#!/usr/bin/env python3
"""Train DEIMv2 on the same splits as train.py / train_rfdetr.py.

    conda activate deimv2
    python scripts/yolo_to_coco.py                        # once, shared with rf-detr
    python scripts/train_deimv2.py --dry-run
    python scripts/train_deimv2.py --arch hgnetv2_n --resolution 1280 \
        --epochs 100 --name deimv2_n_1280

DEIMv2 is a REPO, not a pip package: there is no Python API to call, so this
wrapper generates the run's config (scripts/deimv2_cfg.py), then shells out to
the clone's own train.py. The config is written to the run directory, which is
what makes the run reproducible and what backends.py reads back at eval time.

Comparability rules this script enforces, same as train_rfdetr.py:
  * same splits - it consumes data/merged_coco, which yolo_to_coco.py mirrors
    from merge.py and never re-splits
  * resolution divisible by 32, which every DEIMv2 backbone accepts, so 1280
    matches yolo26n and RFDETRNano/Small exactly
  * every run appends to the SAME results/runs.csv

One protocol difference you cannot configure away: DEIMv2 resizes to a SQUARE
(it stretches, it does not letterbox). See deimv2/README.md - it changes what
the model sees, not what the metric measures.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import deimv2_cfg
from runlog import log_run

# DEIMv2 writes several checkpoints; this is the order we prefer when handing
# one to eval. stg2 is the final (no-aug) stage, stg1 the earlier best.
CKPT_PREFERENCE = ["best_stg2.pth", "best_stg1.pth", "best.pth", "last.pth",
                   "checkpoint.pth"]


def find_checkpoint(out):
    for n in CKPT_PREFERENCE:
        p = Path(out) / n
        if p.exists():
            return p
    return None


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = ap.add_argument_group("what to train")
    g.add_argument("--dataset", default=str(ROOT / "data/merged_coco"))
    g.add_argument("--arch", default="hgnetv2_n",
                   help="hgnetv2_{atto,femto,pico,n,s,m,l,x} | dinov3_{s,m,l,x}. "
                        "hgnetv2_pico (1.5M) / hgnetv2_n (3.6M) are the fair "
                        "size-match for yolo26n (2.57M) on Jetson.")
    g.add_argument("--name", default="deimv2_n_1280")
    g.add_argument("--project", default=str(ROOT / "deimv2/runs"))
    g.add_argument("--tuning", default="",
                   help="fine-tune from a COCO checkpoint (DEIMv2 train.py -t). "
                        "The class head will not match 2 classes and is skipped; "
                        "the backbone/encoder is what you are reusing.")
    g.add_argument("--resume", default="", help="continue a run (train.py -r)")

    g = ap.add_argument_group("schedule")
    g.add_argument("--resolution", type=int, default=1280,
                   help=f"multiple of {deimv2_cfg.RESOLUTION_STEP}; 1280 matches "
                        f"the yolo26n baseline exactly")
    g.add_argument("--epochs", type=int, default=0,
                   help="0 = keep the variant's own tuned `epoches`. Any other "
                        "value ALSO rescales flat_epoch / no_aug_epoch / the "
                        "aug policy epochs, so a short run keeps its shape.")
    g.add_argument("--batch", type=int, default=0,
                   help="TOTAL batch across all GPUs (upstream: total_batch_size). "
                        "0 = the config's own value.")
    g.add_argument("--val-batch", type=int, default=0)
    g.add_argument("--workers", type=int, default=4)
    g.add_argument("--lr", type=float, default=0.0,
                   help="0 = the config's own lr")
    g.add_argument("--no-photometric", action="store_true",
                   help="drop RandomPhotometricDistort. Our hsv_h~0 decision says "
                        "cone hue is signal; this is the DEIMv2 equivalent.")

    g = ap.add_argument_group("hardware")
    g.add_argument("--devices", type=int, default=1, help="GPUs on this node")
    g.add_argument("--device", default="", help="train.py -d, e.g. cuda / cpu")
    g.add_argument("--master-port", default="7777")
    g.add_argument("--no-amp", action="store_true")
    g.add_argument("--seed", type=int, default=0)

    ap.add_argument("--emit-config", default="",
                    help="write the resolved config here and exit")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    home = deimv2_cfg.deimv2_home()
    if not (home / "train.py").exists():
        print(f"[STOP] no DEIMv2 clone at {home}")
        print( "       bash scripts/setup_deimv2_env.sh   (or set $DEIMV2_HOME)")
        return 2

    out = Path(args.project) / args.name
    cfg, meta = deimv2_cfg.build(
        arch=args.arch, dataset=args.dataset, resolution=args.resolution,
        epochs=args.epochs, output_dir=out, batch=args.batch,
        val_batch=args.val_batch, workers=args.workers,
        photometric=not args.no_photometric, seed=args.seed)
    if args.lr:
        cfg.setdefault("optimizer", {})["lr"] = args.lr

    arch = meta["arch"]
    print("resolved config:")
    print(json.dumps({"arch": arch, "upstream": meta["upstream"],
                      "dataset": args.dataset,
                      "num_classes": cfg["num_classes"],
                      "resolution": args.resolution,
                      "epoches": cfg.get("epoches", meta["base_epoches"]),
                      "output_dir": str(out),
                      "devices": args.devices}, indent=2))
    if meta["schedule_notes"]:
        print(f"\n  schedule rescaled from {meta['base_epoches']} epochs:")
        for n in meta["schedule_notes"]:
            print(f"    {n}")
    if args.no_photometric:
        print("\n  RandomPhotometricDistort removed (cone hue is signal)")

    # Backbone weights that DEIMv2 does NOT download for you.
    need = meta["needs_ckpt"]
    if need and not (home / need).exists():
        print(f"\n[STOP] {arch} needs {home / need}, which DEIMv2 does not "
              f"download automatically.")
        print( "       See deimv2/README.md 'Backbone checkpoints'. The hgnetv2_* "
               "variants fetch their backbone themselves.")
        if not args.dry_run:
            return 2

    if args.emit_config:
        print(f"\nwrote {deimv2_cfg.write(args.emit_config, cfg)}")
        return 0
    if args.dry_run:
        import yaml
        print("\n--- config.yml ---")
        print(yaml.safe_dump(cfg, sort_keys=False, width=120))
        return 0

    out.mkdir(parents=True, exist_ok=True)
    cfg_path = deimv2_cfg.write(out / "config.yml", cfg)
    print(f"\n  config -> {cfg_path}")

    # cwd MUST be the clone: upstream configs reference ./ckpts/... relatively,
    # and engine.* is only importable from the repo root.
    if args.devices > 1:
        cmd = ["torchrun", f"--master_port={args.master_port}",
               f"--nproc_per_node={args.devices}", "train.py"]
    else:
        cmd = [sys.executable, "train.py"]
    cmd += ["-c", str(cfg_path), "--seed", str(args.seed)]
    if not args.no_amp:
        cmd += ["--use-amp"]
    if args.device:
        cmd += ["-d", args.device]
    if args.resume:
        rp = Path(args.resume)
        if not rp.exists():
            print(f"[STOP] --resume {rp} does not exist")
            return 2
        cmd += ["-r", str(rp)]
        print(f"[resume] continuing from {rp}")
    elif args.tuning:
        tp = Path(args.tuning)
        if not tp.exists():
            print(f"[STOP] --tuning {tp} does not exist")
            return 2
        cmd += ["-t", str(tp)]
        print(f"[tuning] fine-tuning from {tp}")

    print(f"\n  $ cd {home} && " + " ".join(cmd))
    env = dict(os.environ, DEIMV2_HOME=str(home))
    rc = subprocess.run(cmd, cwd=str(home), env=env).returncode
    print(f"\n[train] exit {rc}")
    if rc != 0:
        return rc

    log_run(run=args.name, notes=args.notes or "deimv2 baseline",
            model=f"deimv2_{arch}", weights="pretrained-default",
            imgsz=args.resolution,
            epochs=cfg.get("epoches", meta["base_epoches"]),
            batch=args.batch or "", device=args.device or "auto",
            dataset="merged_coco")

    best = find_checkpoint(out)
    print(f"\ncheckpoints: {out}")
    if best is None:
        print("  no checkpoint found - check the log above")
        return 0
    print(f"next: python scripts/eval_buckets.py --backend deimv2 --arch {arch} \\")
    print(f"        --weights {best} --imgsz {args.resolution} --run {args.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
