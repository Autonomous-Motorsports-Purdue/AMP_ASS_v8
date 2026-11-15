#!/usr/bin/env python3
"""Per-frame inference latency, measured identically for both model families.

    # in env: cone
    python scripts/bench_latency.py --backend yolo \
        --weights yolo26n/runs/m640_yolo26n_640/weights/best.pt --device mps
    # in env: rfdetr
    python scripts/bench_latency.py --backend rfdetr --arch RFDETRNano \
        --weights rfdetr/runs/.../checkpoint_best_ema.pth --device mps

Reports END-TO-END predict() time: preprocess + forward + postprocess/NMS, i.e.
what a frame actually costs in the loop. Median and p95, because a mean hides
the tail that a real-time system cares about.

Warms up first (MPS/CUDA kernels compile on first call), runs on several
distinct images so nothing is cached, and appends to results/latency.csv.
"""
import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="yolo", choices=["yolo", "rfdetr"])
    ap.add_argument("--arch", default="")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--n", type=int, default=30, help="timed iterations")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    import re
    from PIL import Image
    txt = (ROOT / "configs/cone.yaml").read_text()
    base = Path(re.search(r"^path:\s*(.+)$", txt, re.M).group(1).strip())
    imgs = sorted((base / "val/images").iterdir())[:args.n + args.warmup]
    assert imgs, "no val images"

    def make_item(p):
        w, h = Image.open(p).size
        return {"img": p, "w": w, "h": h}

    # Load the model ONCE. backends.py constructs it per call (fine for a
    # batch eval, fatal for a latency benchmark - it would time model loading,
    # and for rf-detr also optimize_for_inference()'s JIT trace).
    t_load = time.perf_counter()
    if args.backend == "yolo":
        from ultralytics import YOLO
        m = YOLO(args.weights)

        def one(path):
            return m.predict(source=str(path), imgsz=args.imgsz, conf=args.conf,
                             device=args.device, verbose=False)
    else:
        import rfdetr
        cls = getattr(rfdetr, args.arch)
        m = cls(pretrain_weights=args.weights, resolution=args.imgsz)
        if hasattr(m, "optimize_for_inference"):
            try:
                m.optimize_for_inference()
            except Exception as e:
                print(f"  [warn] optimize_for_inference failed: {e}")

        def one(path):
            return m.predict(Image.open(path).convert("RGB"), threshold=args.conf)
    load_ms = (time.perf_counter() - t_load) * 1000

    # warmup - first call compiles kernels / builds the graph
    for p in imgs[:args.warmup]:
        one(p)

    ts = []
    for p in imgs[args.warmup:args.warmup + args.n]:
        t0 = time.perf_counter()
        one(p)
        ts.append((time.perf_counter() - t0) * 1000)

    ts.sort()
    med = statistics.median(ts)
    p95 = ts[int(len(ts) * 0.95) - 1]
    name = args.arch or Path(args.weights).stem
    print(f"\n{'='*70}")
    print(f"  {args.backend}/{name}  imgsz={args.imgsz}  device={args.device}")
    print(f"{'='*70}")
    print(f"  model load   : {load_ms/1000:8.1f} s   (once, EXCLUDED from below)")
    print(f"  frames timed : {len(ts)} (after {args.warmup} warmup)")
    print(f"  median       : {med:8.1f} ms   ({1000/med:6.1f} fps)")
    print(f"  mean         : {statistics.mean(ts):8.1f} ms")
    print(f"  min / p95    : {ts[0]:8.1f} / {p95:.1f} ms")

    f = ROOT / "results/latency.csv"
    new = not f.exists()
    with f.open("a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["backend", "model", "imgsz", "device", "n",
                        "median_ms", "mean_ms", "p95_ms", "fps", "tag"])
        w.writerow([args.backend, name, args.imgsz, args.device, len(ts),
                    round(med, 2), round(statistics.mean(ts), 2), round(p95, 2),
                    round(1000 / med, 2), args.tag])
    print(f"  -> results/latency.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
