#!/usr/bin/env python3
"""data/merged (YOLO) -> data/merged_coco (COCO JSON) for RF-DETR.

RF-DETR reads Roboflow-style COCO, not YOLO txt. This mirrors the EXISTING
splits - it never re-splits - so yolo26n and rf-detr train and are scored on
byte-identical image sets. Re-splitting here would silently invalidate every
comparison.

    python scripts/yolo_to_coco.py            # hardlinks, ~no extra disk
    python scripts/yolo_to_coco.py --copy     # real copies (~5.7 GB)

Note the directory names: RF-DETR expects train/ valid/ test/ - "valid",
not "val". merge.py's split is preserved under the renamed dir.
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from imgmeta import image_size

# Roboflow COCO exports carry a dummy supercategory at id 0 and real classes
# from id 1. RF-DETR is built around that convention.
CATEGORIES = [
    {"id": 0, "name": "cones", "supercategory": "none"},
    {"id": 1, "name": "cone", "supercategory": "cones"},
]
SPLIT_DIR = {"train": "train", "val": "valid", "test": "test"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "configs/cone.yaml"))
    ap.add_argument("--out", default=str(ROOT / "data/merged_coco"))
    ap.add_argument("--copy", action="store_true",
                    help="copy images instead of hardlinking (~5.7 GB)")
    args = ap.parse_args()

    txt = Path(args.data).read_text()
    base = Path(re.search(r"^path:\s*(.+)$", txt, re.M).group(1).strip())
    out = Path(args.out)

    grand = 0
    for split, dname in SPLIT_DIR.items():
        m = re.search(rf"^{split}:\s*(.+)$", txt, re.M)
        if not m:
            print(f"  {split}: not in yaml, skipped")
            continue
        idir = base / m.group(1).strip()
        odir = out / dname
        odir.mkdir(parents=True, exist_ok=True)

        images, annos = [], []
        aid = 1
        for iid, ip in enumerate(sorted(idir.iterdir()), start=1):
            if ip.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                continue
            sz = image_size(ip)
            if sz is None:
                continue
            W, H = sz
            dst = odir / ip.name
            if not dst.exists():
                if args.copy:
                    dst.write_bytes(ip.read_bytes())
                else:
                    try:
                        os.link(ip, dst)
                    except OSError:
                        dst.write_bytes(ip.read_bytes())
            images.append({"id": iid, "file_name": ip.name, "width": W, "height": H})

            lp = Path(str(ip).replace("/images/", "/labels/")).with_suffix(".txt")
            if not lp.exists():
                continue
            for line in lp.read_text().splitlines():
                if not line.strip():
                    continue
                _, cx, cy, bw, bh = (float(x) for x in line.split()[:5])
                bw, bh = bw * W, bh * H
                x, y = cx * W - bw / 2, cy * H - bh / 2
                # clamp: a box nudged outside the frame makes COCO tooling throw
                x, y = max(0.0, x), max(0.0, y)
                bw, bh = min(bw, W - x), min(bh, H - y)
                if bw <= 0 or bh <= 0:
                    continue
                annos.append({"id": aid, "image_id": iid, "category_id": 1,
                              "bbox": [round(x, 2), round(y, 2),
                                       round(bw, 2), round(bh, 2)],
                              "area": round(bw * bh, 2), "iscrowd": 0,
                              "segmentation": []})
                aid += 1

        (odir / "_annotations.coco.json").write_text(json.dumps(
            {"images": images, "annotations": annos, "categories": CATEGORIES}))
        grand += len(annos)
        print(f"  {split:<5} -> {dname:<6} {len(images):>6} images  {len(annos):>7} boxes")

    print(f"\n  total {grand:,} boxes  ->  {out}")
    print(f"  {'hardlinked' if not args.copy else 'copied'}; splits mirror merge.py exactly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
