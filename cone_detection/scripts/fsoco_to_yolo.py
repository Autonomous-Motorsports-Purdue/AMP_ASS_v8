#!/usr/bin/env python3
"""FSOCO Supervisely -> YOLO single-class `cone`.

Facts this encodes (verified from the data):
  - 5 RECTANGLE classes collapse to one class `cone`; the seg_* BITMAP classes
    are a different annotation set and are skipped.
  - points.exterior = [[x1,y1],[x2,y2]] in ABSOLUTE pixels; size = {h, w}.
  - annotation file is <name>.<ext>.json  (extension kept, .jpg and .png mixed)
  - folder = labeling team, filename prefix = SOURCE team. The prefix is the
    split unit, not the folder (amz_* is spread over 20 folders). Prefix case
    varies (BME vs bme), so it is lowercased.
  - resolutions run to 6304x4304; huge images are downscaled on the way out.

Writes a flat YOLO dir plus manifest.csv, which carries everything the split
and the inventory need - including, per image, how many boxes clear the 10px
labeling floor after letterboxing to 640/960/1280.

    python scripts/fsoco_to_yolo.py ~/fsoco_raw -o data/interim/fsoco
"""
import argparse
import collections
import csv
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # 27MP source images are legitimate here

RECT_CLASSES = {"blue_cone", "yellow_cone", "orange_cone",
                "large_orange_cone", "unknown_cone"}
IMG_EXT = {".jpg", ".jpeg", ".png"}
LETTERBOX_SIZES = (640, 960, 1280)
FLOOR_PX = 10.0   # FSOCO labeling floor ~100 sq px; below this, absence != negative


def prefix_of(stem):
    m = re.match(r"([A-Za-z]+)", stem)
    return (m.group(1) if m else "unknown").lower()


def convert_one(job):
    src, folder, out_img_dir, out_lbl_dir, max_side, quality, keep_png = job
    stem = src.stem                       # amz_00000  (src = amz_00000.jpg)
    ann_path = src.parent.parent / "ann" / (src.name + ".json")
    rec = {"folder": folder, "prefix": prefix_of(stem), "src": str(src),
           "status": "ok", "n_boxes": 0, "n_skipped_bitmap": 0,
           "n_degenerate": 0, "n_clamped": 0, "truncated": 0, "knocked_over": 0}

    if not ann_path.exists():
        rec["status"] = "no_ann"
        return rec
    try:
        ann = json.loads(ann_path.read_text())
    except Exception as e:
        rec["status"] = f"bad_json:{type(e).__name__}"
        return rec

    aw, ah = ann["size"]["width"], ann["size"]["height"]
    tags = [t["name"] for t in ann.get("tags", [])]
    rec["tags"] = "|".join(tags)
    rec["has_issue"] = int(any(t.startswith("issue") for t in tags))
    rec["split_tag"] = "test" if "test" in tags else ("train" if "train" in tags else "")

    try:
        im = Image.open(src)
        iw, ih = im.size
    except Exception as e:
        rec["status"] = f"bad_img:{type(e).__name__}"
        return rec
    # The annotation's own size field is the coordinate frame; if it disagrees
    # with the file, normalising by the wrong one silently shifts every box.
    rec["size_mismatch"] = int((iw, ih) != (aw, ah))
    rec["orig_w"], rec["orig_h"] = iw, ih

    boxes = []
    for ob in ann.get("objects", []):
        if ob.get("geometryType") != "rectangle" or ob.get("classTitle") not in RECT_CLASSES:
            rec["n_skipped_bitmap"] += 1
            continue
        for t in ob.get("tags", []):
            if t["name"] in ("truncated", "knocked_over"):
                rec[t["name"]] += 1
        (x1, y1), (x2, y2) = ob["points"]["exterior"]
        x1, x2 = sorted((float(x1), float(x2)))
        y1, y2 = sorted((float(y1), float(y2)))
        cx1, cy1 = max(0.0, x1), max(0.0, y1)
        cx2, cy2 = min(float(aw), x2), min(float(ah), y2)
        if (cx1, cy1, cx2, cy2) != (x1, y1, x2, y2):
            rec["n_clamped"] += 1
        w, h = cx2 - cx1, cy2 - cy1
        if w <= 0 or h <= 0:
            rec["n_degenerate"] += 1
            continue
        boxes.append(((cx1 + cx2) / 2 / aw, (cy1 + cy2) / 2 / ah, w / aw, h / ah))

    rec["n_boxes"] = len(boxes)

    # how many boxes survive the 10px floor once letterboxed to each imgsz
    for s in LETTERBOX_SIZES:
        sc = s / max(aw, ah)
        rec[f"n_ge10_at{s}"] = sum(1 for _, _, _, bh in boxes if bh * ah * sc >= FLOOR_PX)
    rec["n_lt6_at1280"] = sum(1 for _, _, _, bh in boxes
                              if bh * ah * (1280 / max(aw, ah)) < 6.0)

    scale = min(1.0, max_side / max(iw, ih)) if max_side else 1.0
    rec["scale"] = round(scale, 5)
    out_ext = src.suffix.lower() if keep_png else ".jpg"
    if scale < 1.0 and keep_png is False:
        out_ext = ".jpg"
    out_name = f"{folder}__{stem}{out_ext}"
    rec["out"] = out_name

    try:
        if scale < 1.0:
            nw, nh = round(iw * scale), round(ih * scale)
            im.draft("RGB", (nw, nh))          # fast JPEG DCT downscale when possible
            im = im.convert("RGB").resize((nw, nh), Image.LANCZOS)
        elif out_ext in (".jpg", ".jpeg"):
            im = im.convert("RGB")
        rec["out_w"], rec["out_h"] = im.size
        dst = out_img_dir / out_name
        if out_ext in (".jpg", ".jpeg"):
            im.save(dst, "JPEG", quality=quality, subsampling=1, optimize=True)
        else:
            im.save(dst)
    except Exception as e:
        rec["status"] = f"save_fail:{type(e).__name__}"
        return rec
    finally:
        im.close()

    (out_lbl_dir / f"{folder}__{stem}.txt").write_text(
        "".join(f"0 {a:.6f} {b:.6f} {c:.6f} {d:.6f}\n" for a, b, c, d in boxes))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("data/interim/fsoco"))
    ap.add_argument("--max-side", type=int, default=2560,
                    help="downscale so max(w,h) <= this. 2560 = 2x headroom over imgsz 1280")
    ap.add_argument("--quality", type=int, default=92)
    ap.add_argument("--keep-png", action="store_true",
                    help="keep PNGs as PNG (bigger, slower to train on)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--limit", type=int, default=0, help="debug: only N images")
    args = ap.parse_args()

    root = args.root.expanduser().resolve()
    out_img = (args.out / "images").resolve()
    out_lbl = (args.out / "labels").resolve()
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    jobs = []
    for folder in sorted(d.name for d in root.iterdir() if d.is_dir()):
        idir = root / folder / "img"
        if not idir.is_dir():
            continue
        for p in sorted(idir.iterdir()):
            if p.suffix.lower() in IMG_EXT:
                jobs.append((p, folder, out_img, out_lbl,
                             args.max_side, args.quality, args.keep_png))
    if args.limit:
        jobs = jobs[:args.limit]
    print(f"converting {len(jobs):,} images  ->  {args.out}  "
          f"(max_side={args.max_side}, workers={args.workers})", flush=True)

    recs = []
    with ProcessPoolExecutor(args.workers) as ex:
        for i, rec in enumerate(ex.map(convert_one, jobs, chunksize=16), 1):
            recs.append(rec)
            if i % 500 == 0 or i == len(jobs):
                print(f"  {i:>6,}/{len(jobs):,}", flush=True)

    cols = ["out", "folder", "prefix", "status", "split_tag", "tags", "has_issue",
            "orig_w", "orig_h", "out_w", "out_h", "scale", "n_boxes",
            "n_ge10_at640", "n_ge10_at960", "n_ge10_at1280", "n_lt6_at1280",
            "truncated", "knocked_over", "n_skipped_bitmap", "n_degenerate",
            "n_clamped", "size_mismatch", "src"]
    man = args.out / "manifest.csv"
    with man.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in recs:
            w.writerow({k: r.get(k, "") for k in cols})

    okrec = [r for r in recs if r["status"] == "ok"]
    nb = sum(r["n_boxes"] for r in okrec)
    print(f"\n{'='*66}\n  FSOCO -> YOLO\n{'='*66}")
    print(f"  images converted      {len(okrec):>8,} / {len(recs):,}")
    print(f"  boxes written         {nb:>8,}")
    bad = collections.Counter(r["status"] for r in recs if r["status"] != "ok")
    for k, v in bad.items():
        print(f"  ** {k:<18} {v:>8,}")
    for k in ("n_degenerate", "n_clamped", "size_mismatch", "n_skipped_bitmap"):
        t = sum(r.get(k, 0) for r in okrec)
        if t:
            print(f"  {k:<21} {t:>8,}")
    print(f"  images w/ issue tag   {sum(r.get('has_issue',0) for r in okrec):>8,}")
    print(f"  downscaled            {sum(1 for r in okrec if r['scale']<1.0):>8,}")

    print(f"\n  -- boxes clearing the {FLOOR_PX:.0f}px floor after letterbox")
    for s in LETTERBOX_SIZES:
        g = sum(r.get(f"n_ge10_at{s}", 0) for r in okrec)
        print(f"     imgsz {s:<5} {g:>8,}  ({100*g/max(nb,1):5.1f}% usable)")
    lt6 = sum(r.get("n_lt6_at1280", 0) for r in okrec)
    print(f"     <6px @1280  {lt6:>8,}  ({100*lt6/max(nb,1):5.1f}%)")
    print(f"\n  manifest: {man}\n")


if __name__ == "__main__":
    main()
