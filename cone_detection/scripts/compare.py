#!/usr/bin/env python3
"""Matched-protocol comparison: yolo26n vs rf-detr.

Every knob that could hand one model an advantage is set ONCE here and passed
to both. The two models live in different conda envs, so each stage is a
subprocess against the right interpreter.

    python scripts/compare.py --stage all --fraction 0.10 --epochs 5 --imgsz 1344
    python scripts/compare.py --stage report          # re-print from runs.csv
    python scripts/compare.py --dry-run --stage all   # show every command

RESOLUTION PARITY:
  yolo needs a multiple of 32. rf-detr needs a multiple of patch_size *
  num_windows, which is PER VARIANT - Nano/Small/Medium/Large = 32, Base = 56.
  So with Nano or Small, 1280 is already byte-identical to the yolo26n
  baseline; only Base forces LCM(32,56) = 224 (-> 1344). The guard below
  computes this from the chosen arch instead of assuming.

What is held equal: splits (yolo_to_coco mirrors merge.py), resolution, epochs,
--fraction, seed, eval metric (scripts/eval_buckets.py), eval confidence sweep,
IoU threshold, and the images --seed picks for the visual check.

What is NOT equal, and cannot be: optimizer/schedule defaults and pretraining
corpora differ per family. Anyone reading the table should know that.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
def _conda_envs():
    """$CONDA_ENVS, else the envs/ dir this interpreter lives in
    (<base>/envs/cone/bin/python -> <base>/envs), else `conda info --base`."""
    if os.environ.get("CONDA_ENVS"):
        return Path(os.environ["CONDA_ENVS"])
    here = Path(sys.executable).resolve()
    if here.parent.parent.parent.name == "envs":
        return here.parent.parent.parent
    try:
        base = subprocess.run(["conda", "info", "--base"], capture_output=True,
                              text=True, check=True).stdout.strip()
        return Path(base) / "envs"
    except (OSError, subprocess.CalledProcessError):
        return Path("envs")


CONDA = _conda_envs()
DEFAULT_CONF = "0.05,0.10,0.25,0.50"


def run(cmd, dry, label):
    print(f"\n{'='*78}\n  {label}\n{'='*78}")
    print("  $ " + " ".join(str(c) for c in cmd))
    if dry:
        return 0
    return subprocess.run([str(c) for c in cmd]).returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all", "data", "train", "eval", "viz", "report"])
    ap.add_argument("--imgsz", type=int, default=1280,
                    help="must satisfy BOTH models; checked against the chosen "
                         "--rfdetr-arch (Nano/Small need %32, so 1280 works)")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--subset", default="",
                    help="e.g. 010 -> configs/cone_subset_010.yaml + data/subset_010_coco. "
                         "Built by make_subset.py so BOTH models get the identical "
                         "image list; do NOT use yolo --fraction, rf-detr has no equivalent.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--conf", default=DEFAULT_CONF)
    ap.add_argument("--limit", type=int, default=300,
                    help="eval on N val images (0 = all 1,775)")
    ap.add_argument("--n-viz", type=int, default=8)
    ap.add_argument("--viz-conf", type=float, default=0.10)
    ap.add_argument("--topk", type=int, default=25)
    ap.add_argument("--yolo-model", default="yolo26n.pt")
    ap.add_argument("--rfdetr-arch", default="RFDETRSmall")
    ap.add_argument("--yolo-batch", type=int, default=4)
    ap.add_argument("--rfdetr-batch", type=int, default=2)
    ap.add_argument("--rfdetr-grad-accum", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--cone-python", default=str(CONDA / "cone/bin/python"))
    ap.add_argument("--rfdetr-python", default=str(CONDA / "rfdetr/bin/python"))
    ap.add_argument("--tag", default="cmp")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # Parity step is LCM(yolo's 32, this rf-detr variant's patch*windows).
    # It is NOT a constant: Nano/Small/Medium/Large use 32, so 1280 is already
    # exactly matched; only Base's 56 forces 224.


    ypy, rpy = Path(args.cone_python), Path(args.rfdetr_python)
    yname = f"{args.tag}_yolo26n_{args.imgsz}"
    rname = f"{args.tag}_{args.rfdetr_arch.lower()}_{args.imgsz}"
    S = ROOT / "scripts"

    # rfdetr is not importable from THIS interpreter (it lives in the other
    # env), so ask that interpreter rather than guessing.
    from math import gcd
    rstep = 56
    if rpy.exists():
        q = subprocess.run(
            [str(rpy), "-c",
             f"import sys;sys.path.insert(0,{str(ROOT / 'scripts')!r});"
             f"from backends import resolution_step;"
             f"print(resolution_step({args.rfdetr_arch!r}))"],
            capture_output=True, text=True)
        for line in reversed(q.stdout.strip().splitlines()):
            if line.strip().isdigit():
                rstep = int(line.strip()); break
    step = 32 * rstep // gcd(32, rstep)
    if args.imgsz % step:
        lo, hi = (args.imgsz // step) * step, ((args.imgsz // step) + 1) * step
        print(f"[STOP] --imgsz {args.imgsz} breaks resolution parity.")
        print(f"       yolo needs %32, {args.rfdetr_arch} needs %{rstep} -> both need %{step}.")
        print(f"       Use {lo} or {hi}.")
        return 2
    print(f"  resolution parity: {args.imgsz} is divisible by {step} "
          f"(yolo %32, {args.rfdetr_arch} %{rstep}) - byte-identical input size")
    stage, dry = args.stage, args.dry_run
    rc = 0

    if not ypy.exists():
        print(f"[STOP] cone interpreter not found: {ypy}")
        return 2
    if stage in ("all", "train", "eval", "viz") and not rpy.exists():
        print(f"[STOP] rfdetr interpreter not found: {rpy}")
        print( "       run: bash scripts/setup_rfdetr_env.sh")
        return 2

    # ---- data -----------------------------------------------------------
    if stage in ("all", "data"):
        rc |= run([ypy, S / "yolo_to_coco.py"], dry, "COCO mirror of the SAME splits")

    # ---- train ----------------------------------------------------------
    if stage in ("all", "train"):
        ydata = (ROOT / f"configs/cone_subset_{args.subset}.yaml") if args.subset \
                else (ROOT / "configs/cone.yaml")
        cds = (ROOT / f"data/subset_{args.subset}_coco") if args.subset \
              else (ROOT / "data/merged_coco")
        rc |= run([ypy, S / "train.py", "--data", ydata, "--model", args.yolo_model,
                   "--imgsz", args.imgsz, "--epochs", args.epochs,
                   "--fraction", 1.0, "--batch", args.yolo_batch,
                   "--seed", args.seed, "--device", args.device,
                   "--name", yname, "--notes", f"matched compare @{args.imgsz}"],
                  dry, f"TRAIN yolo26n @ {args.imgsz}")
        # rf-detr has no --fraction; it trains on the full COCO train split.
        rc |= run([rpy, S / "train_rfdetr.py", "--arch", args.rfdetr_arch,
                   "--resolution", args.imgsz, "--epochs", args.epochs,
                   "--batch", args.rfdetr_batch,
                   "--dataset", cds, "--grad-accum", args.rfdetr_grad_accum,
                   "--name", rname, "--notes", f"matched compare @{args.imgsz}"],
                  dry, f"TRAIN {args.rfdetr_arch} @ {args.imgsz}")

    ywt = ROOT / f"yolo26n/runs/{yname}/weights/best.pt"
    rwt = ROOT / f"rfdetr/runs/{rname}/checkpoint_best_total.pth"

    # ---- eval -----------------------------------------------------------
    if stage in ("all", "eval"):
        common = ["--imgsz", args.imgsz, "--conf", args.conf, "--split", "val"]
        if args.limit:
            common += ["--limit", args.limit]
        rc |= run([ypy, S / "eval_buckets.py", "--backend", "yolo",
                   "--weights", ywt, "--run", yname, *common],
                  dry, "EVAL yolo26n - bucketed recall")
        rc |= run([rpy, S / "eval_buckets.py", "--backend", "rfdetr",
                   "--arch", args.rfdetr_arch, "--weights", rwt,
                   "--run", rname, *common],
                  dry, f"EVAL {args.rfdetr_arch} - bucketed recall")

    # ---- viz ------------------------------------------------------------
    if stage in ("all", "viz"):
        v = ["--imgsz", args.imgsz, "--n", args.n_viz, "--conf", args.viz_conf,
             "--topk", args.topk, "--seed", args.seed]
        rc |= run([ypy, S / "predict.py", "--backend", "yolo",
                   "--weights", ywt, *v], dry, "VIZ yolo26n (same 8 frames)")
        rc |= run([rpy, S / "predict.py", "--backend", "rfdetr",
                   "--arch", args.rfdetr_arch, "--weights", rwt, *v],
                  dry, f"VIZ {args.rfdetr_arch} (same 8 frames)")

    # ---- report ---------------------------------------------------------
    if stage in ("all", "report") and not dry:
        report(yname, rname, args.imgsz)
    return rc


def report(yname, rname, imgsz):
    import csv
    f = ROOT / "results/runs.csv"
    if not f.exists():
        print("no results/runs.csv yet")
        return
    rows = {}
    for r in csv.DictReader(f.open()):
        if r.get("run") in (yname, rname) and r.get("recall_10_16"):
            rows[r["run"]] = r
    print(f"\n{'='*78}\n  COMPARISON @ imgsz {imgsz}  (identical splits, metric, "
          f"confidence sweep)\n{'='*78}")
    if len(rows) < 2:
        print(f"  need both runs scored; have {sorted(rows) or 'none'}")
        print(f"  (run --stage eval first)")
        return
    cols = [("recall_10_16", "10-16px"), ("recall_16_32", "16-32px"),
            ("recall_32_64", "32-64px"), ("recall_ge64", ">=64px"),
            ("recall", "all"), ("precision", "prec")]
    print(f"  {'run':<30}" + "".join(f"{lab:>10}" for _, lab in cols))
    print(f"  {'-'*30}" + "".join(f"{'-'*9:>10}" for _ in cols))
    for n in (yname, rname):
        r = rows[n]
        print(f"  {n:<30}" + "".join(
            f"{float(r[k]):>10.3f}" if r.get(k) else f"{'-':>10}" for k, _ in cols))
    a, b = rows[yname], rows[rname]
    try:
        d = float(b["recall_10_16"]) - float(a["recall_10_16"])
        print(f"\n  10-16px delta (rf-detr - yolo26n): {d:+.3f}")
        print( "  That bucket decides this comparison - small cones are the objective.")
    except (KeyError, ValueError, TypeError):
        pass


if __name__ == "__main__":
    sys.exit(main())
