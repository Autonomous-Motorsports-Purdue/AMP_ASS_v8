#!/usr/bin/env python3
"""Task 5 - recall bucketed by bbox height, which is the metric that decides things.

Overall mAP is dominated by big near cones that were never at risk of being
missed. The question for obstacle avoidance is "what fraction of the real cones
did we actually find, at the distance where we still have time to steer" - so
this reports RECALL per size bucket at a fixed confidence, and sweeps
confidence so the recall/confidence trade-off per bucket is visible.

Two things it gets right that a naive version gets wrong:

  * Buckets are measured in POST-LETTERBOX pixels at the eval imgsz, not source
    pixels. Source resolutions run from 2872x920 to 6304x4304; a 30px box in a
    6304-wide image and a 30px box in a 640-wide image are completely different
    detection problems, so source-pixel buckets are meaningless.
  * Buckets FLOOR AT 10px. FSOCO's labeling policy is ~100 square pixels, so
    below ~10px the absence of a label does not mean the absence of a cone.
    Sub-floor boxes are reported separately and are NOT scored - they would
    measure the labeling policy, not the model.

    python scripts/eval_buckets.py --weights runs/base_p2_1280/weights/best.pt
    python scripts/eval_buckets.py --weights ... --conf 0.05,0.1,0.25,0.5
    python scripts/eval_buckets.py --self-test      # no model needed
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from runlog import log_run

FLOOR_PX = 10.0
BUCKETS = [(10, 16), (16, 32), (32, 64), (64, math.inf)]
BUCKET_KEYS = ["10_16", "16_32", "32_64", "ge64"]


def bucket_labels():
    out = []
    for lo, hi in BUCKETS:
        out.append(f">={lo:.0f}px" if hi == math.inf else f"{lo:.0f}-{hi:.0f}px")
    return out


def bucket_index(h):
    """Bucket for a post-letterbox height. Returns -1 for sub-floor (unscorable)."""
    if h < FLOOR_PX:
        return -1
    for i, (lo, hi) in enumerate(BUCKETS):
        if lo <= h < hi:
            return i
    return len(BUCKETS) - 1


def letterbox_scale(w, h, imgsz):
    """Ultralytics letterbox scales by min(imgsz/w, imgsz/h) == imgsz/max(w,h)."""
    return imgsz / max(w, h)


def iou_matrix(a, b):
    """a: (N,4) xyxy, b: (M,4) xyxy -> (N,M)"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.clip(aa[:, None] + bb[None, :] - inter, 1e-9, None)


def score(items, confs, iou_thr=0.5):
    """Pure scoring - no model, no I/O, so it can be tested synthetically.

    items: list of dicts with
        gt_boxes  (N,4) xyxy in SOURCE pixels
        gt_bucket (N,)  bucket index, -1 = sub-floor
        pr_boxes  (M,4) xyxy in SOURCE pixels
        pr_conf   (M,)
    Returns {conf: {"recall": [...per bucket], "tp":..., "fp":..., ...}}
    """
    out = {}
    for c in confs:
        n_gt = np.zeros(len(BUCKETS), int)
        n_hit = np.zeros(len(BUCKETS), int)
        sub_gt = sub_hit = 0
        tp = fp = 0
        for it in items:
            gtb, gtk = it["gt_boxes"], it["gt_bucket"]
            keep = it["pr_conf"] >= c if len(it["pr_conf"]) else np.zeros(0, bool)
            prb = it["pr_boxes"][keep]
            prc = it["pr_conf"][keep]
            order = np.argsort(-prc) if len(prc) else np.zeros(0, int)
            prb = prb[order]

            for k in gtk:
                if k < 0:
                    continue
                n_gt[k] += 1
            sub_gt += int((gtk < 0).sum()) if len(gtk) else 0

            M = iou_matrix(prb, gtb)
            taken = np.zeros(len(gtb), bool)
            for pi in range(len(prb)):
                if M.shape[1] == 0:
                    fp += 1
                    continue
                row = M[pi].copy()
                row[taken] = -1
                gi = int(row.argmax())
                if row[gi] >= iou_thr:
                    taken[gi] = True
                    tp += 1
                    # a prediction landing on a sub-floor GT is neither
                    # credited nor penalised: that region is unlabelled-ish
                    if gtk[gi] >= 0:
                        n_hit[gtk[gi]] += 1
                    else:
                        sub_hit += 1
                        tp -= 1
                else:
                    fp += 1
        rec = np.divide(n_hit, np.maximum(n_gt, 1), dtype=float)
        rec[n_gt == 0] = float("nan")
        out[c] = {"recall": rec, "n_gt": n_gt, "n_hit": n_hit,
                  "tp": tp, "fp": fp, "sub_gt": sub_gt, "sub_hit": sub_hit,
                  "precision": tp / max(tp + fp, 1),
                  "recall_all": n_hit.sum() / max(n_gt.sum(), 1)}
    return out


def load_split(data_yaml, split, imgsz):
    """Ground truth + per-image letterbox scale."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from imgmeta import image_size
    import re
    txt = Path(data_yaml).read_text()
    base = Path(re.search(r"^path:\s*(.+)$", txt, re.M).group(1).strip())
    sub = re.search(rf"^{split}:\s*(.+)$", txt, re.M).group(1).strip()
    idir = base / sub
    items = []
    for ip in sorted(idir.iterdir()):
        if ip.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            continue
        lp = Path(str(ip).replace("/images/", "/labels/")).with_suffix(".txt")
        sz = image_size(ip)
        if sz is None:
            continue
        w, h = sz
        sc = letterbox_scale(w, h, imgsz)
        boxes, buckets = [], []
        if lp.exists():
            for line in lp.read_text().splitlines():
                if not line.strip():
                    continue
                _, cx, cy, bw, bh = (float(x) for x in line.split()[:5])
                bw, bh = bw * w, bh * h
                cx, cy = cx * w, cy * h
                boxes.append([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])
                buckets.append(bucket_index(bh * sc))   # POST-LETTERBOX height
        items.append({"img": ip, "w": w, "h": h, "scale": sc,
                      "gt_boxes": np.array(boxes, np.float32).reshape(-1, 4),
                      "gt_bucket": np.array(buckets, int)})
    return items


def run_predictions(items, weights, imgsz, min_conf, device, batch,
                    backend="yolo", arch="", model_config=""):
    """Dispatch to scripts/backends.py. The scoring path below never learns
    which model produced the boxes - that is what makes the models
    comparable."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import backends
    fn = backends.get(backend)
    kw = {}
    if arch:
        kw["arch"] = arch
    if model_config:
        kw["model_config"] = model_config
    return fn(items, weights, imgsz, min_conf, device, batch, **kw)


def report(res, confs, title):
    labs = bucket_labels()
    print(f"\n{'='*78}\n  {title}\n{'='*78}")
    n_gt = res[confs[0]]["n_gt"]
    sub = res[confs[0]]["sub_gt"]
    print(f"  ground-truth boxes per bucket (post-letterbox height):")
    print("   ", "  ".join(f"{l}={n:,}" for l, n in zip(labs, n_gt)))
    print(f"    sub-floor (<{FLOOR_PX:.0f}px, NOT scored): {sub:,}")
    print(f"\n  RECALL by bucket")
    print(f"  {'conf':>6} " + "".join(f"{l:>11}" for l in labs)
          + f"{'all':>9}{'prec':>9}{'FP':>9}")
    for c in confs:
        r = res[c]
        cells = "".join("     ---   " if math.isnan(v) else f"{v:>11.3f}"
                        for v in r["recall"])
        print(f"  {c:>6.3f} {cells}{r['recall_all']:>9.3f}"
              f"{r['precision']:>9.3f}{r['fp']:>9,}")


def self_test():
    """Synthetic end-to-end check of the scoring maths."""
    rng = np.random.default_rng(0)
    imgsz = 1280
    W, H = 2560, 1440                    # scale = 1280/2560 = 0.5
    sc = letterbox_scale(W, H, imgsz)
    assert abs(sc - 0.5) < 1e-9, sc
    # target post-letterbox heights, one per bucket (plus a sub-floor box)
    targets = [6, 12, 24, 48, 200]
    items, expect_counts = [], np.zeros(len(BUCKETS), int)
    for rep in range(40):
        boxes, bks = [], []
        for t in targets:
            src_h = t / sc                       # height in source px
            x = rng.uniform(0, W - 60); y = rng.uniform(0, H - src_h - 1)
            boxes.append([x, y, x + src_h * 0.6, y + src_h])
            k = bucket_index(t)
            bks.append(k)
            if k >= 0:
                expect_counts[k] += 1
        items.append({"gt_boxes": np.array(boxes, np.float32),
                      "gt_bucket": np.array(bks, int)})

    def with_preds(fn):
        out = []
        for ii, it in enumerate(items):
            pb, pc = fn(it, ii) if fn.__code__.co_argcount == 2 else fn(it)
            out.append({**it, "pr_boxes": np.asarray(pb, np.float32).reshape(-1, 4),
                        "pr_conf": np.asarray(pc, np.float32)})
        return out

    ok = True
    def check(name, cond, extra=""):
        nonlocal ok
        print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  ' + extra) if extra else ''}")
        ok &= bool(cond)

    print(f"\n{'='*78}\n  eval_buckets self-test\n{'='*78}")
    print(f"  synthetic: {len(items)} images, heights {targets} post-letterbox px "
          f"(scale {sc})")
    print(f"  expected per-bucket GT counts: "
          f"{dict(zip(bucket_labels(), expect_counts))}")

    # 1. GT fed in as predictions -> recall 1.0 everywhere
    r = score(with_preds(lambda it: (it["gt_boxes"], np.ones(len(it["gt_boxes"])))),
              [0.25])[0.25]
    check("GT-as-predictions gives recall 1.0 in every bucket",
          np.allclose(r["recall"], 1.0), f"recall={np.round(r['recall'],3)}")
    check("GT counts land in the right buckets",
          (r["n_gt"] == expect_counts).all(), f"{r['n_gt']} vs {expect_counts}")
    check("sub-floor boxes excluded from scoring",
          r["sub_gt"] == 40 and r["n_gt"].sum() == 160, f"sub_gt={r['sub_gt']}")
    check("prediction on a sub-floor GT is not counted as a false positive",
          r["fp"] == 0, f"fp={r['fp']}")

    # 2. drop one bucket's boxes -> that bucket 0, others unchanged
    drop = 1   # the 16-32px bucket
    def dropper(it):
        keep = it["gt_bucket"] != drop
        return it["gt_boxes"][keep], np.ones(int(keep.sum()))
    r2 = score(with_preds(dropper), [0.25])[0.25]
    others = [i for i in range(len(BUCKETS)) if i != drop]
    check(f"dropping bucket '{bucket_labels()[drop]}' zeroes only that bucket",
          r2["recall"][drop] == 0.0 and np.allclose(r2["recall"][others], 1.0),
          f"recall={np.round(r2['recall'],3)}")

    # 3. shift boxes so IoU falls below threshold -> recall collapses
    def shifted(it):
        b = it["gt_boxes"].copy()
        b[:, [0, 2]] += (b[:, 2] - b[:, 0])[:, None] * 0.9
        return b, np.ones(len(b))
    r3 = score(with_preds(shifted), [0.25], iou_thr=0.5)[0.25]
    check("0.9-width shift drops recall below 0.05 (IoU gate works)",
          np.nanmax(r3["recall"]) < 0.05, f"max recall={np.nanmax(r3['recall']):.3f}")

    # 4. confidence sweep monotonicity
    def graded(it):
        # confidence 0.9 for big boxes, 0.2 for small ones
        h = it["gt_boxes"][:, 3] - it["gt_boxes"][:, 1]
        return it["gt_boxes"], np.where(h * 0.5 >= 32, 0.9, 0.2)
    rs = score(with_preds(graded), [0.1, 0.5])
    lo, hi = rs[0.1]["recall"], rs[0.5]["recall"]
    check("raising conf 0.1->0.5 cannot increase recall in any bucket",
          np.all(np.nan_to_num(hi, nan=0) <= np.nan_to_num(lo, nan=0) + 1e-9),
          f"{np.round(lo,2)} -> {np.round(hi,2)}")
    check("raising conf kills exactly the low-confidence small buckets",
          hi[0] == 0.0 and hi[1] == 0.0 and hi[2] == 1.0 and hi[3] == 1.0,
          f"hi={np.round(hi,3)}")

    # 5. half the boxes in one bucket -> recall exactly 0.5
    # each image holds exactly ONE box per bucket, so "every other box in this
    # bucket" means every other IMAGE, not every other index within an image
    def half(it, ii):
        b, k = it["gt_boxes"], it["gt_bucket"]
        sel = ~((k == 2) & (ii % 2 == 0))
        return b[sel], np.ones(int(sel.sum()))
    r5 = score(with_preds(half), [0.25])[0.25]
    check("removing the 32-64px box from every other image gives recall 0.5 there",
          abs(r5["recall"][2] - 0.5) < 1e-9, f"recall={np.round(r5['recall'],3)}")

    print(f"\n  {'ALL SELF-TESTS PASSED' if ok else 'SELF-TESTS FAILED'}\n")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="")
    ap.add_argument("--data", default=str(ROOT / "configs/cone.yaml"))
    ap.add_argument("--split", default="val", choices=["val", "test", "train"])
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", default="0.05,0.10,0.25,0.50",
                    help="comma list to sweep; recall-vs-confidence per bucket")
    ap.add_argument("--log-conf", type=float, default=0.25,
                    help="which confidence gets written to results/runs.csv")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--run", default="", help="run name for results/runs.csv")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--backend", default="yolo",
                    choices=["yolo", "rfdetr", "deimv2"],
                    help="which model family produced --weights")
    ap.add_argument("--arch", default="",
                    help="rfdetr: RFDETRNano|RFDETRSmall|RFDETRBase|RFDETRMedium|RFDETRLarge. "
                         "deimv2: hgnetv2_{atto,femto,pico,n,s,m,l,x}|dinov3_{s,m,l,x}")
    ap.add_argument("--model-config", default="",
                    help="deimv2 only: the run's config.yml. Defaults to the one "
                         "train_deimv2.py wrote next to --weights.")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if not args.weights:
        ap.error("--weights is required (or use --self-test)")

    confs = sorted(float(c) for c in args.conf.split(","))
    device = args.device
    if device == "auto":
        import torch
        device = "0" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")

    items = load_split(args.data, args.split, args.imgsz)
    if args.limit:
        items = items[:args.limit]
    print(f"{args.split}: {len(items):,} images, "
          f"{sum(len(i['gt_boxes']) for i in items):,} gt boxes, imgsz {args.imgsz}")
    items = run_predictions(items, args.weights, args.imgsz, min(confs), device,
                            args.batch, backend=args.backend, arch=args.arch,
                            model_config=args.model_config)

    res = score(items, confs, args.iou)
    report(res, confs, f"{Path(args.weights).parent.parent.name}  [{args.split}]  imgsz {args.imgsz}")

    out = ROOT / "results" / f"buckets_{args.run or Path(args.weights).parent.parent.name}_{args.split}.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["conf"] + [f"recall_{k}" for k in BUCKET_KEYS]
                   + [f"n_gt_{k}" for k in BUCKET_KEYS]
                   + ["recall_all", "precision", "tp", "fp", "sub_floor_gt"])
        for c in confs:
            r = res[c]
            w.writerow([c] + [("" if math.isnan(v) else round(float(v), 5)) for v in r["recall"]]
                       + list(map(int, r["n_gt"]))
                       + [round(r["recall_all"], 5), round(r["precision"], 5),
                          r["tp"], r["fp"], r["sub_gt"]])
    print(f"\n  wrote {out}")

    lc = min(confs, key=lambda c: abs(c - args.log_conf))
    r = res[lc]
    log_run(run=args.run or Path(args.weights).parent.parent.name,
            notes=f"eval {args.split} @conf{lc}", weights=str(args.weights),
            imgsz=args.imgsz, device=device, conf_thresh=lc, iou_thresh=args.iou,
            recall_10_16=round(float(r["recall"][0]), 5),
            recall_16_32=round(float(r["recall"][1]), 5),
            recall_32_64=round(float(r["recall"][2]), 5),
            recall_ge64=round(float(r["recall"][3]), 5),
            n_10_16=int(r["n_gt"][0]), n_16_32=int(r["n_gt"][1]),
            n_32_64=int(r["n_gt"][2]), n_ge64=int(r["n_gt"][3]),
            recall=round(float(r["recall_all"]), 5),
            precision=round(float(r["precision"]), 5))
    print(f"  logged to results/runs.csv @conf {lc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
