#!/usr/bin/env python3
"""Task 1 - inventory a YOLO-format dataset.

Reports image/label counts, the literal class names as they appear on disk,
image dimensions, and a histogram of bbox heights in PIXELS (the number that
actually decides whether a cone is detectable at range).

Usage:
    python3 scripts/inventory.py data/raw/fsoco --name fsoco
    python3 scripts/inventory.py data/raw/aks   --name aks

Stdlib only - runs before ultralytics/torch are installed.
"""
import argparse
import collections
import concurrent.futures as cf
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from imgmeta import image_size

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# The buckets the project actually cares about (task 5 reuses these).
HEIGHT_BUCKETS = [(0, 10), (10, 25), (25, 60), (60, 10**9)]
BUCKET_LABELS = ["<10px", "10-25px", "25-60px", ">60px"]


def bucket_of(h):
    for i, (lo, hi) in enumerate(HEIGHT_BUCKETS):
        if lo <= h < hi:
            return i
    return len(HEIGHT_BUCKETS) - 1


def find_class_names(root):
    """Pull class names from any data.yaml / *.yaml without requiring pyyaml."""
    out = {}
    for y in list(root.rglob("data.yaml")) + list(root.rglob("*.yaml")):
        try:
            text = y.read_text(errors="replace")
        except OSError:
            continue
        # flow style:  names: ['cone', '1']   or   names: [cone, 1]
        m = re.search(r"^names\s*:\s*\[(.*?)\]", text, re.M | re.S)
        if m:
            items = [s.strip().strip("'\"") for s in m.group(1).split(",") if s.strip()]
            out[str(y.relative_to(root))] = {i: n for i, n in enumerate(items)}
            continue
        # block style:  names:\n  0: cone\n  1: '1'
        m = re.search(r"^names\s*:\s*$(.*?)(?=^\S|\Z)", text, re.M | re.S)
        if m:
            d = {}
            for line in m.group(1).splitlines():
                mm = re.match(r"\s*(\d+)\s*:\s*(.+?)\s*$", line)
                if mm:
                    d[int(mm.group(1))] = mm.group(2).strip().strip("'\"")
                else:
                    mm = re.match(r"\s*-\s*(.+?)\s*$", line)
                    if mm:
                        d[len(d)] = mm.group(1).strip().strip("'\"")
            if d:
                out[str(y.relative_to(root))] = d
    return out


def label_for_image(img_path):
    """images/foo/bar.jpg -> labels/foo/bar.txt (the ultralytics convention)."""
    parts = list(img_path.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts).with_suffix(".txt")
    return img_path.with_suffix(".txt")


def parse_label(path):
    """Return (rows, n_malformed). rows = [(cls, cx, cy, w, h), ...]"""
    rows, bad = [], 0
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return rows, bad
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        p = line.split()
        if len(p) < 5:
            bad += 1
            continue
        try:
            # >5 fields means segmentation polygons; flag rather than guess
            if len(p) > 5:
                bad += 1
                continue
            rows.append((int(float(p[0])), *(float(x) for x in p[1:5])))
        except ValueError:
            bad += 1
    return rows, bad


def scan_one(img_path):
    lbl = label_for_image(img_path)
    size = image_size(img_path)
    has_lbl = lbl.exists()
    rows, bad = parse_label(lbl) if has_lbl else ([], 0)
    return img_path, size, has_lbl, rows, bad


def split_of(p, root):
    rel = p.relative_to(root).parts
    for s in ("train", "valid", "val", "test"):
        if s in rel:
            return "val" if s == "valid" else s
    return "(none)"


def prefix_of(name):
    """Leading token of a filename - surfaces the gfr_/bme_ FSOCO-derived images."""
    m = re.match(r"([A-Za-z]+)[_\-]", name)
    return m.group(1).lower() if m else "(no prefix)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--name", required=True, help="short dataset name, used for the output file")
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        sys.exit(f"not a directory: {root}")

    imgs = [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT and p.is_file()]
    all_txt = [p for p in root.rglob("*.txt") if p.name.lower() not in
               {"readme.txt", "classes.txt", "requirements.txt", "notes.txt"}]

    print(f"\n{'='*68}\n  {args.name}  <-  {root}\n{'='*68}")
    if not imgs:
        print("  NO IMAGES FOUND.")
        if list(root.rglob("ann/*.json")) or list(root.rglob("*/ann")):
            print("  Looks like Supervisely format (ann/*.json), not YOLO.")
            print("  FSOCO ships Supervisely by default - convert before merging.")
        return

    names_by_yaml = find_class_names(root)

    dims = collections.Counter()
    cls_counts = collections.Counter()
    boxes_per_img = collections.Counter()
    heights = []
    bucket_counts = collections.Counter()
    split_imgs = collections.Counter()
    prefixes = collections.Counter()
    n_with_label = n_empty_label = n_no_label = n_bad = n_nosize = 0
    degenerate = 0
    out_of_range = 0

    with cf.ThreadPoolExecutor(args.workers) as ex:
        for img, size, has_lbl, rows, bad in ex.map(scan_one, imgs):
            split_imgs[split_of(img, root)] += 1
            prefixes[prefix_of(img.name)] += 1
            n_bad += bad
            if size:
                dims[size] += 1
            else:
                n_nosize += 1
            if not has_lbl:
                n_no_label += 1
                continue
            if not rows:
                n_empty_label += 1
                continue
            n_with_label += 1
            boxes_per_img[len(rows)] += 1
            for cls, cx, cy, w, h in rows:
                cls_counts[cls] += 1
                if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0 < w <= 1.0 and 0 < h <= 1.0):
                    out_of_range += 1
                if w <= 0 or h <= 0:
                    degenerate += 1
                    continue
                if size:
                    hp = h * size[1]
                    heights.append(hp)
                    bucket_counts[bucket_of(hp)] += 1

    n_boxes = sum(cls_counts.values())
    print(f"\n-- counts")
    print(f"  images on disk          {len(imgs):>8,}")
    print(f"  label .txt files        {len(all_txt):>8,}")
    print(f"  images w/ >=1 box       {n_with_label:>8,}")
    print(f"  images w/ empty label   {n_empty_label:>8,}   (backgrounds)")
    print(f"  images w/ NO label file {n_no_label:>8,}")
    print(f"  bounding boxes total    {n_boxes:>8,}")
    if n_with_label:
        print(f"  boxes per labelled img  {n_boxes/n_with_label:>8.1f}  avg"
              f"   max {max(boxes_per_img) if boxes_per_img else 0}")
    if n_bad:
        print(f"  ** malformed/polygon label rows: {n_bad:,}")
    if degenerate:
        print(f"  ** zero-area boxes: {degenerate:,}")
    if out_of_range:
        print(f"  ** boxes with coords outside [0,1]: {out_of_range:,}")
    if n_nosize:
        print(f"  ** images whose dims could not be read: {n_nosize:,}")

    print(f"\n-- splits (by path)")
    for s, c in split_imgs.most_common():
        print(f"  {s:<10} {c:>8,}")

    print(f"\n-- class names declared in yaml")
    if names_by_yaml:
        for y, d in names_by_yaml.items():
            print(f"  {y}")
            for i in sorted(d):
                print(f"      {i}: {d[i]!r}")
    else:
        print("  (no yaml found)")

    print(f"\n-- class ids actually present in labels")
    any_names = next(iter(names_by_yaml.values()), {})
    for cls, c in sorted(cls_counts.items()):
        nm = any_names.get(cls, "?")
        print(f"  id {cls:<3} {nm!r:<22} {c:>8,}  ({100*c/max(n_boxes,1):5.1f}%)")

    print(f"\n-- image dimensions (top 12 of {len(dims)} distinct)")
    for (w, h), c in dims.most_common(12):
        print(f"  {w:>5} x {h:<5} {c:>8,}  ({100*c/max(len(imgs),1):5.1f}%)")

    print(f"\n-- filename prefixes (top 12) [overlap hunting]")
    for p, c in prefixes.most_common(12):
        print(f"  {p:<16} {c:>8,}")

    print(f"\n-- bbox HEIGHT in pixels  (n={len(heights):,})")
    if heights:
        sh = sorted(heights)
        def pct(q):
            return sh[min(len(sh) - 1, int(q * len(sh)))]
        print(f"  min {sh[0]:.1f}   p05 {pct(.05):.1f}   p25 {pct(.25):.1f}   "
              f"median {pct(.5):.1f}   p75 {pct(.75):.1f}   p95 {pct(.95):.1f}   max {sh[-1]:.1f}")
        print()
        for i, lbl in enumerate(BUCKET_LABELS):
            c = bucket_counts[i]
            bar = "#" * int(48 * c / max(n_boxes, 1))
            print(f"  {lbl:<9} {c:>8,}  ({100*c/max(n_boxes,1):5.1f}%) {bar}")
        print()
        edges = [0, 5, 10, 15, 20, 25, 35, 50, 60, 80, 120, 200, 400, 10**9]
        fine = collections.Counter()
        for h in heights:
            for j in range(len(edges) - 1):
                if edges[j] <= h < edges[j + 1]:
                    fine[j] += 1
                    break
        for j in range(len(edges) - 1):
            hi = "inf" if edges[j + 1] > 10**8 else str(edges[j + 1])
            c = fine[j]
            bar = "#" * int(48 * c / max(len(heights), 1))
            print(f"  {edges[j]:>4}-{hi:<5} {c:>8,} {bar}")

    args.out.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": args.name,
        "root": str(root),
        "images": len(imgs),
        "label_files": len(all_txt),
        "images_with_boxes": n_with_label,
        "images_empty_label": n_empty_label,
        "images_no_label_file": n_no_label,
        "boxes": n_boxes,
        "malformed_rows": n_bad,
        "degenerate_boxes": degenerate,
        "coords_out_of_range": out_of_range,
        "yaml_names": {k: {str(i): n for i, n in v.items()} for k, v in names_by_yaml.items()},
        "class_counts": {str(k): v for k, v in sorted(cls_counts.items())},
        "splits": dict(split_imgs),
        "dims_top": [{"w": w, "h": h, "n": c} for (w, h), c in dims.most_common(30)],
        "n_distinct_dims": len(dims),
        "prefixes": dict(prefixes.most_common(40)),
        "height_buckets": {BUCKET_LABELS[i]: bucket_counts[i] for i in range(len(BUCKET_LABELS))},
        "height_pct": (lambda s: {} if not s else {
            "min": s[0], "p05": s[int(.05*len(s))], "p25": s[int(.25*len(s))],
            "median": s[int(.5*len(s))], "p75": s[int(.75*len(s))],
            "p95": s[int(.95*len(s))], "max": s[-1]})(sorted(heights)),
    }
    dst = args.out / f"inventory_{args.name}.json"
    dst.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {dst}\n")


if __name__ == "__main__":
    main()
