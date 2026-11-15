#!/usr/bin/env python3
"""Inference + visual check for ANY model family, through backends.py.

    # yolo26n (env: cone)
    python scripts/predict.py --backend yolo \
        --weights yolo26n/runs/local_640_f10/weights/best.pt \
        --imgsz 640 --n 8 --conf 0.10 --topk 25

    # rf-detr (env: rfdetr)
    python scripts/predict.py --backend rfdetr --arch RFDETRSmall \
        --weights rfdetr/runs/rfdetr_s_1280/checkpoint_best_total.pth \
        --imgsz 1280 --n 8 --conf 0.10 --topk 25

    # deimv2 (env: deimv2)
    python scripts/predict.py --backend deimv2 --arch hgnetv2_n \
        --weights deimv2/runs/deimv2_n_1280/best_stg2.pth \
        --imgsz 1280 --n 8 --conf 0.10 --topk 25

    # labels only, no model
    python scripts/predict.py --gt-only --n 8

--seed picks the SAME images for every backend, so the folders are
directly comparable frame by frame.

On thresholds: --conf 0.01 shows a detector's low-confidence tail, which looks
like noise for ANY model and hides real differences. Judge at 0.10+, or use
--topk to see only what the model actually believes.
"""
import argparse
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

GREEN, RED = (0, 220, 0), (255, 40, 40)


def load_gt(lp, W, H):
    out = []
    if not lp.exists():
        return out
    for line in lp.read_text().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        _, cx, cy, w, h = (float(x) for x in p[:5])
        out.append(((cx - w / 2) * W, (cy - h / 2) * H,
                    (cx + w / 2) * W, (cy + h / 2) * H))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="yolo",
                    choices=["yolo", "rfdetr", "deimv2"])
    ap.add_argument("--arch", default="",
                    help="rfdetr: e.g. RFDETRSmall.  deimv2: e.g. hgnetv2_n")
    ap.add_argument("--model-config", default="",
                    help="deimv2 only: the run's config.yml (default: next to --weights)")
    ap.add_argument("--weights", default="")
    ap.add_argument("--data", default=str(ROOT / "configs/cone.yaml"))
    ap.add_argument("--split", default="val", choices=["val", "test", "train"])
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--topk", type=int, default=0,
                    help="draw only the N highest-confidence preds")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--gt-only", action="store_true")
    ap.add_argument("--seed", type=int, default=0, help="same seed = same images")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import re
    txt = Path(args.data).read_text()
    base = Path(re.search(r"^path:\s*(.+)$", txt, re.M).group(1).strip())
    idir = base / re.search(rf"^{args.split}:\s*(.+)$", txt, re.M).group(1).strip()
    imgs = sorted(p for p in idir.iterdir()
                  if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    random.Random(args.seed).shuffle(imgs)
    imgs = imgs[:args.n]

    items = []
    for ip in imgs:
        w, h = Image.open(ip).size
        items.append({"img": ip, "w": w, "h": h,
                      "pr_boxes": np.zeros((0, 4), np.float32),
                      "pr_conf": np.zeros(0, np.float32)})

    if args.weights and not args.gt_only:
        import backends
        kw = {"arch": args.arch} if args.arch else {}
        if args.model_config:
            kw["model_config"] = args.model_config
        items = backends.get(args.backend)(items, args.weights, args.imgsz,
                                           args.conf, args.device, 1, **kw)

    tag = args.backend if args.weights and not args.gt_only else "gt_only"
    name = Path(args.weights).stem if args.weights else "labels"
    home = ROOT / {"rfdetr": "rfdetr", "deimv2": "deimv2"}.get(
        args.backend, "yolo26n") / "viz"
    out = Path(args.out) if args.out else home / f"{tag}_{name}_{args.split}"
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*78}")
    print(f"  {tag}  [{args.split}]  imgsz={args.imgsz}  conf>={args.conf}"
          + (f"  topk={args.topk}" if args.topk else ""))
    print(f"  weights: {args.weights or '(none)'}")
    print(f"{'='*78}")
    print(f"  {'image':<32} {'GT':>5} {'drawn':>6} {'best':>7}  {'largest pred':>14}")
    print(f"  {'-'*32} {'-'*5} {'-'*6} {'-'*7}  {'-'*14}")

    tg = tp = 0
    for it in items:
        ip = it["img"]
        im = Image.open(ip).convert("RGB")
        W, H = im.size
        gt = load_gt(Path(str(ip).replace("/images/", "/labels/")).with_suffix(".txt"), W, H)
        d = ImageDraw.Draw(im)
        for b in gt:
            d.rectangle(list(b), outline=GREEN, width=2)

        order = np.argsort(-it["pr_conf"]) if len(it["pr_conf"]) else []
        order = order[:args.topk] if args.topk else order
        for i in order:
            x1, y1, x2, y2 = it["pr_boxes"][i]
            d.rectangle([x1, y1, x2, y2], outline=RED, width=2)
            d.text((x1 + 2, max(0, y1 - 11)), f"{it['pr_conf'][i]:.2f}", fill=RED)

        im.save(out / ip.name, quality=90)
        best = float(it["pr_conf"].max()) if len(it["pr_conf"]) else 0.0
        if len(order):
            k = max(order, key=lambda i: it["pr_boxes"][i][3] - it["pr_boxes"][i][1])
            b = it["pr_boxes"][k]
            dims = f"{int(b[2]-b[0])}x{int(b[3]-b[1])}"
        else:
            dims = "-"
        tg += len(gt); tp += len(order)
        print(f"  {ip.name[:32]:<32} {len(gt):>5} {len(order):>6} {best:>7.3f}  {dims:>14}")

    print(f"  {'-'*32} {'-'*5} {'-'*6}")
    print(f"  {'TOTAL':<32} {tg:>5} {tp:>6}")
    print(f"\n  green = ground truth   red = prediction")
    print(f"  wrote {len(items)} images to {out}")
    if sys.platform == "darwin":
        subprocess.run(["open", str(out)], check=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
