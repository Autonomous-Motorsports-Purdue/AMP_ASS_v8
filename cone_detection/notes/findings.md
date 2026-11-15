# Findings — tasks 1 & 2

## Task 1 — inventory

### FSOCO (converted Supervisely → YOLO, `data/interim/fsoco/`)

| | |
|---|---|
| images | 11,572 |
| boxes | **220,862** |
| classes | 5 rectangle classes → collapsed to `cone` |
| bitmap `seg_*` classes | skipped (different annotation set) |
| conversion errors | **0** — no degenerate, no clamped, no size mismatch |
| disk | 24 GB → 5.5 GB (max side 2560, JPEG q92) |

Converter verified by round-tripping boxes back to source coords: **max error
0.0021 px**.

Reproduces the verified figures exactly — 220,862 boxes, and usable-box share
after letterbox 30.6 / 47.1 / 60.7% at 640 / 960 / 1280, `<6px@1280` = 14.2%.

Box height at stored resolution: median 22px, p05 8.9px.
`<10px` 9.6% · `10–25px` 47.2% · `25–60px` 27.6% · `>60px` 15.5%

**Issue tags:** 98 images carry **114 issue tags** (59 "label all cones…" +
45 "inaccurate bounding boxes" + 10 "missing bounding boxes") — some images
carry more than one. All 98 are in folder `prom`. → **exclude `prom` from
val/test.**

**Splits:** all 11,572 tagged `train`. No `test` tag exists. We build our own.

**Split unit is the filename prefix, lowercased** — 17 distinct prefixes.
`amz` alone is 3,943 images (**34% of the dataset**) spread across 20 folders,
so it dominates whichever side of the split it lands on.

### AKS (`data/raw/aks/`, Roboflow YOLOv11 export)

| | |
|---|---|
| images | 6,948 (**5,301 unique stems** — 737 stems have Roboflow augmentation copies) |
| boxes | 19,511 |
| classes | `names: ['1', 'cone']` → **id 0 = `'1'`, id 1 = `cone`** |
| splits | train 4,919 / val 1,361 / test 668 |

⚠️ **The class order is reversed from intuition.** In the label files `0` is the
junk-looking class and `1` is `cone`. Assuming `0 == cone` silently inverts the
whole dataset.

## Task 2 — what is the AKS class `1`?

**Not junk. Real cones — do not drop them.**

All 7 class-`1` boxes live in a single image, `bme_00625`. Rendering it shows a
track with cones down both edges: the annotator labeled the **blue** left-hand
line as class `1` and the **yellow** right-hand line as `cone`. The two lines
mirror each other in size almost exactly. This is FSOCO's `blue_cone` /
`yellow_cone` split leaking through the AKS conversion.

→ **Merge class `1` into `cone`.** Dropping it would plant 7 unlabeled cones in
a training image — false-negative supervision, the exact error we can least
afford.

## The big one — AKS labels only near cones

2,997 AKS images (56.5%) stem-match an FSOCO image. Comparing box counts on
those matched pairs:

| | |
|---|---|
| identical box count | 84 |
| **AKS has FEWER boxes** | **2,913** — **66,238 boxes missing** |
| AKS has more | 0 |

e.g. `amz_01945`: AKS 4 boxes, FSOCO 64.

The missing ones are the small far cones. On matched images:

| | n | p10 | p25 | median | p75 | p90 |
|---|---|---|---|---|---|---|
| FSOCO | 14,168 | 11 | 15 | **25** | 50 | 103 |
| AKS | 1,658 | 20 | 34 | **67** | 122 | 207 |

57.8% of AKS boxes are in the **top quarter by size** within their own image;
median size-rank 0.200 (0 = biggest cone in frame).

The native (non-FSOCO) AKS images are skewed even harder:

| group | imgs | boxes | median h | `<25px` |
|---|---|---|---|---|
| native-AKS | 3,951 | 11,279 | 72px | **5.1%** |
| fsoco-derived AKS | 2,997 | 8,232 | 63px | 17.3% |
| FSOCO | 11,572 | 220,862 | 22px | ~57% |

**Why this matters more than the dedupe:** mixing these naively gives the model
contradictory supervision — FSOCO says "that 12px blob is a cone", AKS says the
same blob is background. For a project whose whole objective is recall on small
distant cones, AKS labels as-is actively train that objective away.

Consequences for task 3:
1. For stem-matched images, **keep the FSOCO copy and its labels, drop the AKS
   copy.** This is dedupe with a rule, not a coin flip.
2. **val/test must come from completely-labeled data (FSOCO).** Measuring recall
   against incomplete labels measures nothing.
3. Native-AKS is the only **on-domain** data we have (karting, not Formula
   Student) — valuable for exactly that reason, and under-labeled. Open
   decision, see README.

---

# Task 3 — merge

`scripts/merge.py` → `data/merged/` (hardlinked, ~no extra disk)

| split | images | boxes | fsoco | aks | prefixes |
|---|---|---|---|---|---|
| train | 10,296 | 158,128 | 7,992 | 2,304 | amz, dhen, eco, ecurieaix, eufs, frt, fsbizkaia, gfr, prom, pwrrt, tuwr, ugent, ulm + aks:* |
| val | 1,775 | 34,488 | 1,775 | 0 | bme, ff |
| test | 1,776 | 34,701 | 1,776 | 0 | ka, mms |
| **total** | **13,847** | **227,317** | | | |

Verified: no prefix overlap between any two splits, no duplicate image names,
zero AKS images in val/test, zero issue-tagged images outside train, only class
id `0` present, and `check_det_dataset` loads it.

## Dedupe results

| how | n | action |
|---|---|---|
| stem-match AKS↔FSOCO | 2,997 | AKS copy dropped, FSOCO kept |
| phash AKS↔FSOCO beyond stem-match | **0** | — |
| FSOCO-internal true duplicates | 29 | dropped |
| FSOCO-internal phash collisions | 593 | **kept** (adjacent video frames) |

Two things worth knowing:

**phash found zero cross-dataset matches that stem-matching missed.** Roboflow
preserved the original filenames, so the overlap was fully visible by name. The
hash pass is still worth keeping — it's what proves there is no hidden overlap.

**64-bit phash cannot distinguish adjacent video frames.** The first run dropped
622 FSOCO images as "exact duplicates"; sampling 12 of them showed *none* were
duplicates — they were consecutive frames with different labels (one pair had
3 boxes vs 9). That would have thrown away ~10,600 real boxes. The internal pass
now treats a phash match as a *candidate* and confirms with mean pixel
difference (<1.0/255 on a 64×64 thumbnail). 29 survived that test.

They also could not have leaked: adjacent frames share a prefix, and the split
is by prefix.

## Split rationale

- **`amz` (34% of FSOCO) and `prom` (all 98 issue-tagged images) forced to train.**
  `amz` is too large to sit in val/test without unbalancing it.
- val = `bme` + `ff` (1,775 imgs, 12.8%), test = `ka` + `mms` (1,776, 12.8%).
  Four distinct source teams, no overlap with train.
- Native-AKS prefixes are namespaced `aks:*` so they can never collide with an
  FSOCO team prefix. Before this, hex-named AKS files (`ff1e2b80…`) produced a
  junk prefix `ff` that collided with the real FSOCO team `ff_*` and looked
  exactly like split leakage. `prefix_of` now requires a `_`/`-` separator.

## AKS class remap

`names: ['1','cone']` → **both** id 0 and id 1 map to `cone`, per task 2.
