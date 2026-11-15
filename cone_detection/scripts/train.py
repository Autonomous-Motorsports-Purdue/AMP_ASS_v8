#!/usr/bin/env python3
"""Task 4 - train a cone detector. Same script locally and on a cluster GPU.

Nothing about the machine is baked in: --data, --model, --imgsz, --batch and
--device are all arguments, and --device auto picks cuda > mps > cpu. Every
run appends its config to results/runs.csv (task 5 fills in the bucketed
recall for the same run name).

Defaults encode the decisions in README.md:
  model yolo26n-p2.yaml, imgsz 1280, close_mosaic 10, fliplr on, hsv_h 0.

    # cluster
    python scripts/train.py --device 0 --batch -1 --name base_p2_1280
    # multi-GPU
    python scripts/train.py --device 0,1,2,3 --batch 32 --name base_p2_1280
    # see the resolved config without training
    python scripts/train.py --dry-run
    # quick smoke test on 2% of the data
    python scripts/train.py --fraction 0.02 --epochs 1 --name smoke

NOTE: imgsz 1280 with the P2 head does NOT train on Apple MPS - see the
negative-results section of the README.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from runlog import log_run


def pick_device(requested):
    import torch
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = ap.add_argument_group("what to train")
    g.add_argument("--data", default=str(ROOT / "configs/cone.yaml"))
    g.add_argument("--model", default="yolo26n-p2.yaml",
                   help="yolo26n.pt | yolo26n-p2.yaml | yolo11n.pt | path to .pt")
    g.add_argument("--weights", default="",
                   help="pretrained weights to load into a .yaml architecture")
    g.add_argument("--name", default="base_p2_1280", help="run name; also the runs.csv key")
    g.add_argument("--project", default=str(ROOT / "yolo26n/runs"),
                   help="per-model workspace; rf-detr uses rfdetr/runs")

    g = ap.add_argument_group("machine")
    g.add_argument("--device", default="auto", help="auto | cpu | mps | 0 | 0,1,2,3")
    g.add_argument("--batch", type=int, default=-1,
                   help="-1 = ultralytics AutoBatch (CUDA only); set explicitly elsewhere")
    g.add_argument("--workers", type=int, default=8)

    g = ap.add_argument_group("schedule")
    g.add_argument("--imgsz", type=int, default=1280)
    g.add_argument("--epochs", type=int, default=100)
    g.add_argument("--close-mosaic", type=int, default=10)
    g.add_argument("--patience", type=int, default=100)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--fraction", type=float, default=1.0,
                   help="train on a RANDOM subset (built by make_subset.py). NOT "
                        "ultralytics' --fraction, which takes a contiguous slice of "
                        "the sorted file list - on this prefix-sorted dataset that "
                        "yields aks-only data. See the traps section.")
    g.add_argument("--resume", action="store_true")

    g = ap.add_argument_group("augmentation")
    g.add_argument("--fliplr", type=float, default=0.5, help="keep: a cone is symmetric")
    g.add_argument("--hsv-h", type=float, default=0.0, help="keep ~0: cone hue is signal")
    g.add_argument("--hsv-s", type=float, default=0.7)
    g.add_argument("--hsv-v", type=float, default=0.4)
    g.add_argument("--mosaic", type=float, default=1.0)
    g.add_argument("--scale", type=float, default=0.5)
    g.add_argument("--rect", action="store_true",
                   help="queued experiment: most common source res is 2872x920 (3.1:1)")

    ap.add_argument("--dry-run", action="store_true", help="print the resolved config and exit")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    # --fraction: ultralytics implements this as im_files[:count] - a contiguous
    # slice of the SORTED list, no shuffle. On this dataset the sort puts aks__*
    # first, so --fraction 0.10 trains on 1 team (aks, incomplete labels) instead
    # of 34. Measured: mAP50 0.2531 that way vs 0.5662 on a random sample of the
    # same size. So build a real random subset and hand ultralytics the whole of it.
    real_fraction = args.fraction
    if args.fraction < 1.0:
        from make_subset import build_subset
        sub_yaml, _ = build_subset(args.data, args.fraction, args.seed, quiet=True)
        print(f"[fraction] random subset -> {Path(sub_yaml).name} "
              f"(NOT ultralytics' contiguous slice)")
        args.data = str(sub_yaml)
        args.fraction = 1.0

    device = pick_device(args.device)
    if args.batch == -1 and device in ("cpu", "mps"):
        print(f"[warn] AutoBatch (--batch -1) is CUDA-only; falling back to 4 on {device}")
        args.batch = 4

    cfg = dict(
        data=args.data, model=args.model, imgsz=args.imgsz, epochs=args.epochs,
        batch=args.batch, device=device, workers=args.workers,
        close_mosaic=args.close_mosaic, patience=args.patience, seed=args.seed,
        fraction=args.fraction, fliplr=args.fliplr, hsv_h=args.hsv_h,
        hsv_s=args.hsv_s, hsv_v=args.hsv_v, mosaic=args.mosaic, scale=args.scale,
        rect=args.rect, project=args.project, name=args.name,
    )
    print("resolved config:")
    print(json.dumps(cfg, indent=2))
    if args.imgsz >= 1280 and "p2" in args.model.lower() and device == "mps":
        print("\n[STOP] imgsz>=1280 with a P2 head does not train on MPS "
              "(see README negative results). Use a CUDA device.")
        if not args.dry_run:
            return 2
    if args.dry_run:
        return 0

    from ultralytics import YOLO
    if args.resume:
        # ultralytics only resumes from a checkpoint: YOLO("x.yaml").train(resume=True)
        # warns "not a resumable training checkpoint" and silently restarts at
        # epoch 0 in the same run dir. So resume means: load this run's last.pt.
        last = Path(args.project) / args.name / "weights/last.pt"
        if not last.exists():
            print(f"[STOP] --resume but no checkpoint at {last}")
            return 2
        print(f"[resume] {last}")
        m = YOLO(str(last))
    elif args.weights:
        m = YOLO(args.model).load(args.weights)
    else:
        m = YOLO(args.model)

    r = m.train(exist_ok=True, plots=True, val=True, resume=args.resume, **{
        k: v for k, v in cfg.items() if k not in ("model",)})

    box = r.box
    log_run(run=args.name,
            notes=(args.notes or "task 4 baseline")
                  + (f" | random subset fraction={real_fraction}"
                     if real_fraction < 1.0 else ""),
            model=args.model, weights=args.weights or "pretrained-default",
            imgsz=args.imgsz, epochs=args.epochs, batch=args.batch,
            close_mosaic=args.close_mosaic, fliplr=args.fliplr, hsv_h=args.hsv_h,
            hsv_s=args.hsv_s, hsv_v=args.hsv_v, mosaic=args.mosaic,
            scale=args.scale, device=device, dataset=Path(args.data).name,
            map50=round(float(box.map50), 5), map50_95=round(float(box.map), 5),
            precision=round(float(box.mp), 5), recall=round(float(box.mr), 5))
    print(f"\nweights: {r.save_dir}/weights/best.pt")
    print(f"logged to results/runs.csv as run={args.name}")
    print(f"next: python scripts/eval_buckets.py --weights {r.save_dir}/weights/best.pt "
          f"--run {args.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
