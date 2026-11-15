#!/usr/bin/env python3
"""Build a subsampled training set in BOTH layouts from ONE sampled list.

Why this exists: train.py has --fraction, rf-detr does not. Using YOLO's
--fraction would hand rf-detr the full 10,296 images while yolo26n saw 1,030 -
a 10x data advantage that would read as an architecture difference. This
samples ONE list of images and materialises it as YOLO *and* COCO, so both
models train on a byte-identical subset.

val and test are NEVER subsampled - you always score against the full split.

    python scripts/make_subset.py --fraction 0.10 --seed 0
    -> data/subset_010/            (YOLO)   + configs/cone_subset_010.yaml
    -> data/subset_010_coco/       (COCO)

Then:
    train.py      --data configs/cone_subset_010.yaml --fraction 1.0
    train_rfdetr.py --dataset data/subset_010_coco
"""
import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from imgmeta import image_size

CATEGORIES = [{"id": 0, "name": "cones", "supercategory": "none"},
              {"id": 1, "name": "cone", "supercategory": "cones"}]
COCO_DIR = {"train": "train", "val": "valid", "test": "test"}


def link(src, dst):
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        dst.write_bytes(src.read_bytes())


def build_subset(data_yaml=None, fraction=0.10, seed=0, quiet=False):
    """Materialise a random subset in YOLO + COCO from ONE sampled list.

    Returns (yolo_yaml_path, coco_dir). Reuses an existing subset with the
    same tag instead of rebuilding.
    """
    args = argparse.Namespace(
        data=str(data_yaml or (ROOT / "configs/cone.yaml")),
        fraction=fraction, seed=seed)
    return _build(args, quiet=quiet)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "configs/cone.yaml"))
    ap.add_argument("--fraction", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    _build(args)
    return 0


def _build(args, quiet=False):
    tag = f"{int(round(args.fraction * 100)):03d}"
    yroot = ROOT / f"data/subset_{tag}"
    croot = ROOT / f"data/subset_{tag}_coco"

    txt = Path(args.data).read_text()
    base = Path(re.search(r"^path:\s*(.+)$", txt, re.M).group(1).strip())

    yaml_p = ROOT / f"configs/cone_subset_{tag}.yaml"
    if yaml_p.exists() and (croot / "train/_annotations.coco.json").exists():
        if not quiet:
            print(f"  reusing existing subset {tag} ({yaml_p.name})")
        return yaml_p, croot

    if not quiet:
        print(f"\n  fraction={args.fraction}  seed={args.seed}  tag={tag}")
    for split in ("train", "val", "test"):
        m = re.search(rf"^{split}:\s*(.+)$", txt, re.M)
        if not m:
            continue
        idir = base / m.group(1).strip()
        imgs = sorted(p for p in idir.iterdir()
                      if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        if split == "train" and args.fraction < 1.0:
            # deterministic: identical list for both models
            rnd = random.Random(args.seed)
            imgs = sorted(rnd.sample(imgs, max(1, int(len(imgs) * args.fraction))))
            note = f"sampled {len(imgs)}"
        else:
            note = f"FULL {len(imgs)}"

        yi, yl = yroot / split / "images", yroot / split / "labels"
        cd = croot / COCO_DIR[split]
        for d in (yi, yl, cd):
            d.mkdir(parents=True, exist_ok=True)

        images, annos, aid = [], [], 1
        for iid, ip in enumerate(imgs, start=1):
            sz = image_size(ip)
            if sz is None:
                continue
            W, H = sz
            lp = Path(str(ip).replace("/images/", "/labels/")).with_suffix(".txt")
            link(ip, yi / ip.name)
            link(ip, cd / ip.name)
            if lp.exists():
                link(lp, yl / lp.name)
            images.append({"id": iid, "file_name": ip.name, "width": W, "height": H})
            if not lp.exists():
                continue
            for line in lp.read_text().splitlines():
                if not line.strip():
                    continue
                _, cx, cy, bw, bh = (float(x) for x in line.split()[:5])
                bw, bh = bw * W, bh * H
                x, y = max(0.0, cx * W - bw / 2), max(0.0, cy * H - bh / 2)
                bw, bh = min(bw, W - x), min(bh, H - y)
                if bw <= 0 or bh <= 0:
                    continue
                annos.append({"id": aid, "image_id": iid, "category_id": 1,
                              "bbox": [round(x, 2), round(y, 2), round(bw, 2), round(bh, 2)],
                              "area": round(bw * bh, 2), "iscrowd": 0, "segmentation": []})
                aid += 1
        (cd / "_annotations.coco.json").write_text(json.dumps(
            {"images": images, "annotations": annos, "categories": CATEGORIES}))
        if not quiet:
            print(f"  {split:<5} {note:<16} {len(annos):>7} boxes  -> {yi.parent.name}/ + {cd.name}/")

    yaml_p.write_text(
        f"# WRITTEN BY scripts/make_subset.py (fraction={args.fraction} seed={args.seed})\n"
        f"# Identical image list to data/subset_{tag}_coco - do not hand-edit.\n"
        f"path: {yroot}\ntrain: train/images\nval: val/images\ntest: test/images\n\n"
        f"nc: 1\nnames:\n  0: cone\n")
    if not quiet:
        print(f"\n  wrote {yaml_p}")
        print(f"  yolo   : train.py --data {yaml_p.name} --fraction 1.0")
        print(f"  rf-detr: train_rfdetr.py --dataset data/subset_{tag}_coco")
    return yaml_p, croot


if __name__ == "__main__":
    sys.exit(main())
