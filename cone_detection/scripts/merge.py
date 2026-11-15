#!/usr/bin/env python3
"""Task 3 - merge FSOCO + AKS into one single-class dataset.

Encodes the decisions from notes/findings.md:

  * Single class `cone`. AKS ships `names: ['1','cone']`, so BOTH id 0 and id 1
    map to cone - class `1` is blue cones, not junk (see task 2).
  * FSOCO WINS every duplicate. AKS is missing 66,238 boxes on the images it
    shares with FSOCO because it labels only large near cones; keeping the AKS
    copy would teach the model that small far cones are background.
  * Duplicate detection is stem-match OR perceptual hash - Roboflow re-encodes
    and renames (`bme_00625_jpg.rf.<hash>.jpg`), so neither alone is enough.
  * Split by LOWERCASED FILENAME PREFIX (= source team), never randomly and
    never by folder: folder is the labeling team and `amz_*` spans 20 folders.
    Adjacent video frames are near-duplicates, so a random split leaks val
    into train.
  * val/test are FSOCO-only - measuring recall against AKS's incomplete labels
    measures nothing.
  * Native-AKS (no FSOCO match) is train-only: it is our only karting-domain
    data, but its labels stop at ~25px.
  * Folder `prom` holds all 98 issue-tagged images -> train only.

    python scripts/merge.py
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

import imagehash
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from imgmeta import image_size

Image.MAX_IMAGE_PIXELS = None
ROOT = Path(__file__).resolve().parent.parent
RF_PAT = re.compile(r"^(.*?)_(jpg|png|jpeg)\.rf\.[0-9a-f]+$")
POP = np.array([bin(i).count("1") for i in range(256)], np.uint8)


def prefix_of(stem):
    """FSOCO source-team prefix, e.g. amz_00123 -> 'amz'.

    The separator is required. Without it, hex/UUID filenames (native AKS is
    full of them) yield junk split units like 'ff' or 'ca' - and 'ff' collides
    with the real FSOCO team `ff_*`, which would look like split leakage.
    """
    m = re.match(r"([A-Za-z]+)[_-]", stem)
    return (m.group(1) if m else "unknown").lower()


def phash_one(path):
    """64-bit perceptual hash as 8 uint8s.

    Uses imagehash's reference implementation rather than a hand-rolled DCT -
    this hash decides which images get deleted, and a hand-rolled version
    scored only 12 bits apart on *unrelated* images, which is no margin at all.
    draft() keeps JPEG decode cheap on the 27MP originals.
    """
    try:
        im = Image.open(path)
        im.draft("L", (64, 64))
        h = imagehash.phash(im.convert("L"))
        return np.packbits(h.hash.flatten())
    except Exception:
        return None


def read_boxes(p, remap_all_to_zero=True, min_px=0.0, img_h=None):
    out, dropped = [], 0
    for line in Path(p).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        f = line.split()
        if len(f) != 5:
            continue
        try:
            cx, cy, w, h = (float(x) for x in f[1:])
        except ValueError:
            continue
        cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
        w, h = min(w, 1.0), min(h, 1.0)
        if w <= 0 or h <= 0:
            dropped += 1
            continue
        if min_px and img_h and h * img_h < min_px:
            dropped += 1
            continue
        out.append((0, cx, cy, w, h))
    return out, dropped


def link_or_copy(src, dst):
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)            # hardlink: instant, no extra disk
    except OSError:
        import shutil
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fsoco", type=Path, default=ROOT / "data/interim/fsoco")
    ap.add_argument("--aks", type=Path, default=ROOT / "data/raw/aks")
    ap.add_argument("--out", type=Path, default=ROOT / "data/merged")
    ap.add_argument("--val-prefixes", default="", help="comma list; default = auto")
    ap.add_argument("--test-prefixes", default="")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--phash-thresh", type=int, default=6,
                    help="hamming distance <= this counts as the same image")
    ap.add_argument("--min-box-px", type=float, default=0.0,
                    help="drop boxes shorter than this at imgsz (queued experiment: 6)")
    ap.add_argument("--imgsz", type=int, default=1280, help="frame for --min-box-px")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()

    # ---------------------------------------------------------------- gather
    fman = {r["out"]: r for r in csv.DictReader(open(args.fsoco / "manifest.csv"))
            if r["status"] == "ok"}
    fsoco = []
    for out, r in fman.items():
        stem = Path(r["src"]).stem
        fsoco.append({
            "src_img": args.fsoco / "images" / out,
            "src_lbl": args.fsoco / "labels" / (Path(out).stem + ".txt"),
            "stem": stem, "prefix": r["prefix"], "folder": r["folder"],
            "dataset": "fsoco", "has_issue": int(r["has_issue"]),
            "h": int(r["out_h"]), "w": int(r["out_w"]), "name": Path(out).stem,
        })
    fstems = {d["stem"] for d in fsoco}

    aks, seen_stem = [], {}
    for split in ("train", "valid", "test"):
        for lp in sorted((args.aks / split / "labels").glob("*.txt")):
            ip = next((args.aks / split / "images").glob(lp.stem + ".*"), None)
            if ip is None:
                continue
            m = RF_PAT.match(lp.stem)
            stem = m.group(1) if m else lp.stem
            if stem in seen_stem:          # Roboflow augmentation copy
                seen_stem[stem] += 1
                continue
            seen_stem[stem] = 1
            aks.append({
                "src_img": ip, "src_lbl": lp, "stem": stem,
                "prefix": "aks:" + prefix_of(stem), "folder": f"aks/{split}",
                "dataset": "aks", "has_issue": 0, "h": None, "w": None,
                "name": "aks__" + stem,
                "fsoco_derived": stem in fstems,
            })
    aug_dropped = sum(v - 1 for v in seen_stem.values() if v > 1)
    print(f"FSOCO {len(fsoco):,} images | AKS {len(aks):,} unique stems "
          f"({aug_dropped:,} Roboflow augmentation copies skipped)")

    # ------------------------------------------------------------ dedupe
    print(f"\nhashing {len(fsoco)+len(aks):,} images ({args.workers} workers)...")
    with ProcessPoolExecutor(args.workers) as ex:
        fh = list(ex.map(phash_one, [d["src_img"] for d in fsoco], chunksize=32))
        ah = list(ex.map(phash_one, [d["src_img"] for d in aks], chunksize=32))

    fidx = [i for i, h in enumerate(fh) if h is not None]
    F = np.array([fh[i] for i in fidx], np.uint8) if fidx else np.zeros((0, 8), np.uint8)

    collisions = []
    drop_aks = set()
    for i, d in enumerate(aks):
        if d["fsoco_derived"]:
            drop_aks.add(i)
            collisions.append((d["stem"], "stem-match", "", 0))
    # phash pass for the ones the stem match missed
    cand = [i for i in range(len(aks)) if i not in drop_aks and ah[i] is not None]
    if cand and len(F):
        A = np.array([ah[i] for i in cand], np.uint8)
        for s in range(0, len(A), 256):
            blk = A[s:s + 256]
            dist = POP[blk[:, None, :] ^ F[None, :, :]].sum(-1)
            best = dist.argmin(1)
            for r in range(len(blk)):
                if dist[r, best[r]] <= args.phash_thresh:
                    ai = cand[s + r]
                    drop_aks.add(ai)
                    collisions.append((aks[ai]["stem"], "phash",
                                       fsoco[fidx[best[r]]]["stem"], int(dist[r, best[r]])))
    # Exact-duplicate pass inside FSOCO. A matching phash is only a CANDIDATE
    # here: 64-bit phash cannot tell adjacent video frames apart, and FSOCO is
    # video-derived. Confirming with actual pixels showed 0/12 sampled phash
    # collisions were true duplicates - they were consecutive frames carrying
    # different labels (one pair 3 boxes vs 9). Those are real training data,
    # they share a prefix so they cannot leak across the split, and dropping
    # them cost ~6,400 boxes. So: verify before deleting.
    seen_h, fdrop = {}, set()
    cand_pairs = []
    for n, i in enumerate(fidx):
        key = F[n].tobytes()
        if key in seen_h:
            cand_pairs.append((i, seen_h[key]))
        else:
            seen_h[key] = i

    def thumb(path):
        im = Image.open(path)
        im.draft("L", (64, 64))
        return np.asarray(im.convert("L").resize((64, 64), Image.LANCZOS), np.float32)

    _tc = {}
    for i, j in cand_pairs:
        for k in (i, j):
            if k not in _tc:
                try:
                    _tc[k] = thumb(fsoco[k]["src_img"])
                except Exception:
                    _tc[k] = None
        a, b = _tc[i], _tc[j]
        if a is None or b is None:
            continue
        mad = float(np.abs(a - b).mean())
        if mad < 1.0:                       # genuinely the same picture
            fdrop.add(i)
            collisions.append((fsoco[i]["stem"], "fsoco-internal-duplicate",
                               fsoco[j]["stem"], 0))
        else:
            collisions.append((fsoco[i]["stem"], "phash-collision-KEPT(adjacent frame)",
                               fsoco[j]["stem"], round(mad, 1)))
    print(f"  FSOCO phash collisions              : {len(cand_pairs):,} "
          f"({len(fdrop):,} true duplicates, {len(cand_pairs)-len(fdrop):,} adjacent frames kept)")
    print(f"  AKS dropped as duplicates of FSOCO : {len(drop_aks):,}")
    print(f"  FSOCO internal exact duplicates     : {len(fdrop):,}")

    fsoco = [d for i, d in enumerate(fsoco) if i not in fdrop]
    aks_keep = [d for i, d in enumerate(aks) if i not in drop_aks]
    native = [d for d in aks_keep if not d["fsoco_derived"]]
    print(f"  native-AKS kept (train only)        : {len(native):,}")

    # ------------------------------------------------------------- split
    # `prom` carries every issue-tagged image -> train only.
    by_pref = collections.Counter(d["prefix"] for d in fsoco)
    total = sum(by_pref.values())
    forced_train = {"prom"}
    # amz is ~34% of FSOCO; too big to sit in val/test without unbalancing it
    biggest = max(by_pref, key=by_pref.get)
    if by_pref[biggest] / total > 0.25:
        forced_train.add(biggest)

    if args.val_prefixes or args.test_prefixes:
        val_p = {p.strip().lower() for p in args.val_prefixes.split(",") if p.strip()}
        test_p = {p.strip().lower() for p in args.test_prefixes.split(",") if p.strip()}
    else:
        pool = sorted((p for p in by_pref if p not in forced_train),
                      key=lambda p: -by_pref[p])
        val_p, test_p = set(), set()
        nv = nt = 0
        tv, tt = args.val_frac * total, args.test_frac * total
        for p in pool:
            if nv >= tv and nt >= tt:
                break
            if (nv / tv if tv else 1) <= (nt / tt if tt else 1):
                val_p.add(p); nv += by_pref[p]
            else:
                test_p.add(p); nt += by_pref[p]

    def split_of(d):
        if d["dataset"] == "aks":
            return "train"
        if d["prefix"] in val_p:
            return "val"
        if d["prefix"] in test_p:
            return "test"
        return "train"

    print(f"\nsplit by prefix (forced to train: {sorted(forced_train)})")
    print(f"  val  prefixes: {sorted(val_p)}")
    print(f"  test prefixes: {sorted(test_p)}")

    # ------------------------------------------------------------- write
    for s in ("train", "val", "test"):
        (args.out / s / "images").mkdir(parents=True, exist_ok=True)
        (args.out / s / "labels").mkdir(parents=True, exist_ok=True)

    stats = collections.defaultdict(lambda: collections.Counter())
    manifest = []
    dropped_small = 0
    for d in fsoco + native:
        s = split_of(d)
        # a box's height at training time is  h_norm * img_h * (imgsz / max(img_w, img_h))
        lb_h = None
        if args.min_box_px:
            if d["h"] is None:
                sz = image_size(d["src_img"])
                d["w"], d["h"] = sz if sz else (None, None)
            if d["h"]:
                lb_h = d["h"] * (args.imgsz / max(d["w"], d["h"]))
        boxes, dr = read_boxes(d["src_lbl"], min_px=args.min_box_px, img_h=lb_h)
        dropped_small += dr
        img_dst = args.out / s / "images" / (d["name"] + d["src_img"].suffix)
        link_or_copy(d["src_img"], img_dst)
        (args.out / s / "labels" / (d["name"] + ".txt")).write_text(
            "".join(f"0 {a:.6f} {b:.6f} {c:.6f} {e:.6f}\n" for _, a, b, c, e in boxes))
        stats[s]["images"] += 1
        stats[s]["boxes"] += len(boxes)
        stats[s][d["dataset"]] += 1
        manifest.append({"split": s, "name": d["name"], "dataset": d["dataset"],
                         "prefix": d["prefix"], "folder": d["folder"],
                         "n_boxes": len(boxes), "has_issue": d["has_issue"]})

    with (args.out / "manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        w.writeheader(); w.writerows(manifest)
    with (args.out / "dedupe_collisions.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["aks_or_fsoco_stem", "how", "matched_fsoco_stem", "hamming"])
        w.writerows(collisions)

    yaml = ROOT / "configs" / "cone.yaml"
    yaml.write_text(
        "# WRITTEN BY scripts/merge.py - do not hand-edit.\n"
        "# `path` is absolute on purpose: ultralytics resolves a relative path\n"
        "# against the CWD, not against this file.\n"
        f"path: {args.out.resolve()}\n"
        "train: train/images\nval: val/images\ntest: test/images\n\nnc: 1\nnames:\n  0: cone\n")

    print(f"\n{'='*68}\n  merged -> {args.out}\n{'='*68}")
    print(f"  {'split':<8}{'images':>9}{'boxes':>10}{'fsoco':>9}{'aks':>8}  prefixes")
    for s in ("train", "val", "test"):
        pl = sorted({m["prefix"] for m in manifest if m["split"] == s and m["dataset"] == "fsoco"})
        print(f"  {s:<8}{stats[s]['images']:>9,}{stats[s]['boxes']:>10,}"
              f"{stats[s]['fsoco']:>9,}{stats[s]['aks']:>8,}  {','.join(pl)[:52]}")
    tot = sum(stats[s]["images"] for s in ("train", "val", "test"))
    print(f"  {'TOTAL':<8}{tot:>9,}{sum(stats[s]['boxes'] for s in ('train','val','test')):>10,}")
    if dropped_small:
        print(f"\n  boxes dropped by --min-box-px {args.min_box_px}: {dropped_small:,}")
    print(f"\n  dedupe collisions logged: {len(collisions):,} -> {args.out/'dedupe_collisions.csv'}")
    print(f"  dataset yaml rewritten  : {yaml}\n")


if __name__ == "__main__":
    main()
