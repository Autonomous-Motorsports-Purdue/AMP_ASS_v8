#!/usr/bin/env python3
"""Train RF-DETR on the same splits as train.py, for a like-for-like compare.

    conda activate rfdetr
    python scripts/yolo_to_coco.py                       # once
    python scripts/train_rfdetr.py --dry-run
    python scripts/train_rfdetr.py --arch RFDETRSmall --resolution 1280 \
        --epochs 100 --name rfdetr_s_1280

Comparability rules this script enforces:
  * same splits (yolo_to_coco.py mirrors merge.py; it never re-splits)
  * resolution divisible by patch_size * num_windows, which is PER VARIANT:
    Nano/Small/Medium/Large = 32 (so 1280 matches yolo26n exactly), Base = 56.
  * every run appends to the SAME results/runs.csv as the YOLO runs.

RF-DETR is a set predictor: no NMS, no duplicate suppression to tune. It has
a fixed query budget (~300); this dataset peaks at 122 boxes in one image, so
there is headroom.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from runlog import log_run


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = ap.add_argument_group("what to train")
    g.add_argument("--dataset", default=str(ROOT / "data/merged_coco"))
    g.add_argument("--arch", default="RFDETRSmall",
                   help="RFDETRNano|RFDETRSmall|RFDETRBase|RFDETRMedium|RFDETRLarge. "
                        "Small/Nano are the fair size-match for yolo26n on Jetson.")
    g.add_argument("--weights", default="", help="resume / fine-tune from a checkpoint")
    g.add_argument("--name", default="rfdetr_s_1280")
    g.add_argument("--project", default=str(ROOT / "rfdetr/runs"))

    g = ap.add_argument_group("schedule")
    g.add_argument("--resolution", type=int, default=1280, help="must be divisible by patch_size*num_windows for --arch "
                        "(Nano/Small/Medium/Large=32, Base=56); checked at runtime")
    g.add_argument("--epochs", type=int, default=100)
    g.add_argument("--batch", type=int, default=4)
    g.add_argument("--grad-accum", type=int, default=4,
                   help="effective batch = batch * grad_accum")
    g.add_argument("--lr", type=float, default=1e-4)
    g.add_argument("--workers", type=int, default=8)
    g.add_argument("--device", default="auto")
    g.add_argument("--devices", type=int, default=1, help="GPUs on this node")
    g.add_argument("--num-nodes", type=int, default=1)
    g.add_argument("--strategy", default="auto", help="auto | ddp | ddp_find_unused_parameters_true")
    g.add_argument("--accelerator", default="auto", help="auto | gpu | cpu | mps")
    g.add_argument("--checkpoint-interval", type=int, default=1,
                   help="epochs between checkpoints; 1 so a walltime kill loses <=1 epoch")
    g.add_argument("--resume", default="", help="path to last.ckpt to continue from")
    g.add_argument("--early-stopping", action="store_true")
    g.add_argument("--patience", type=int, default=10)

    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT / "scripts"))
    from backends import resolution_step
    step = resolution_step(args.arch)
    if args.resolution % step:
        near = max(step, round(args.resolution / step) * step)
        print(f"[STOP] {args.arch} needs --resolution divisible by {step} "
              f"(patch_size * num_windows); {args.resolution} is not.")
        print(f"       Nearest valid: {near}.")
        return 2

    ds = Path(args.dataset)
    missing = [d for d in ("train", "valid") if not (ds / d / "_annotations.coco.json").exists()]
    if missing:
        print(f"[STOP] {ds} missing {missing}. Run: python scripts/yolo_to_coco.py")
        return 2

    out = Path(args.project) / args.name
    cfg = dict(arch=args.arch, dataset=str(ds), resolution=args.resolution,
               epochs=args.epochs, batch=args.batch, grad_accum=args.grad_accum,
               lr=args.lr, workers=args.workers, device=args.device,
               output_dir=str(out), weights=args.weights or "pretrained-default")
    print("resolved config:")
    print(json.dumps(cfg, indent=2))
    print(f"\n  effective batch = {args.batch} x {args.grad_accum} = "
          f"{args.batch * args.grad_accum}")
    if args.dry_run:
        return 0

    import rfdetr
    cls = getattr(rfdetr, args.arch, None)
    if cls is None:
        avail = [n for n in dir(rfdetr) if n.startswith("RFDETR")]
        print(f"[STOP] unknown --arch {args.arch!r}; available: {avail}")
        return 2

    kw = {"resolution": args.resolution}
    if args.weights:
        kw["pretrain_weights"] = args.weights
    m = cls(**kw)

    out.mkdir(parents=True, exist_ok=True)
    tkw = dict(dataset_dir=str(ds), epochs=args.epochs, batch_size=args.batch,
               grad_accum_steps=args.grad_accum, lr=args.lr,
               num_workers=args.workers, output_dir=str(out),
               devices=args.devices, num_nodes=args.num_nodes,
               strategy=args.strategy, accelerator=args.accelerator,
               checkpoint_interval=args.checkpoint_interval,
               early_stopping=args.early_stopping,
               early_stopping_patience=args.patience,
               tensorboard=True)
    if args.device != "auto":
        tkw["device"] = args.device      # absorbed by train(), mapped to the PTL accelerator
    if args.resume:
        rp = Path(args.resume)
        if not rp.exists():
            print(f"[STOP] --resume {rp} does not exist")
            return 2
        tkw["resume"] = str(rp)
        print(f"[resume] continuing from {rp}")
    m.train(**tkw)

    log_run(run=args.name, notes=args.notes or "rf-detr baseline",
            model=args.arch, weights=args.weights or "pretrained-default",
            imgsz=args.resolution, epochs=args.epochs, batch=args.batch,
            device=args.device, dataset="merged_coco")
    print(f"\ncheckpoints: {out}")
    print(f"next: python scripts/eval_buckets.py --backend rfdetr --arch {args.arch} \\")
    print(f"        --weights {out}/checkpoint_best_total.pth \\")
    print(f"        --imgsz {args.resolution} --run {args.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
