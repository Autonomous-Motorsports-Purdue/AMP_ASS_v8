# Cone obstacle detection — Autonomous Karting Series

> In AMP_ASS_v8 this folder is the training and evaluation project behind
> `parts/cone_detector.py` (issue #4). Its history lives in
> [nyaniv/aks-cone-detection](https://github.com/nyaniv/aks-cone-detection).

Detect traffic cones as **mid-track obstacles to steer around**.

The track is unbounded (road bleeds into grass), so cones here are *not*
boundary markers. That is the one assumption that shapes everything else:
a cone is an object to avoid, its color carries no meaning, and the cone that
matters most is the small, far-away one we have not reached yet.

- **Model:** Ultralytics YOLO26n with a P2 head, at imgsz 1280
- **Target:** Jetson Orin Nano Super, TensorRT FP16 engine
- **Classes:** one — `cone`. No color distinction.
- **Deployable model:** `models/base_p2_1280.pt` (6 MB, tracked in git)

## What we optimize

**Recall, bucketed by bbox height in pixels.** A missed cone is a collision;
a false positive is a downstream problem we filter temporally. Overall mAP is
dominated by big near cones that were never in danger of being missed, so it is
logged for context and otherwise ignored.

Buckets: `10-16` · `16-32` · `32-64` · `>=64` px, post-letterbox (see Eval).

Corollary: we do **not** buy precision with a high confidence threshold.
False positives get handled by temporal filtering in the planner, not here.

## Results

`models/base_p2_1280.pt`: yolo26n-p2, imgsz 1280, 50 epochs from COCO-pretrained
`yolo26n.pt`, trained on a Colab T4. Validation set: 1,775 FSOCO images from
two teams (`bme`, `ff`) never seen in training. IoU 0.5.

| conf | 10-16 px | 16-32 px | 32-64 px | >=64 px | precision | false positives |
|---|---|---|---|---|---|---|
| 0.05 | 0.961 | 0.979 | 0.986 | 0.996 | 0.393 | 35,848 |
| 0.10 | 0.953 | 0.976 | 0.986 | 0.995 | 0.502 | 22,874 |
| **0.25** | **0.934** | **0.967** | **0.981** | **0.993** | 0.711 | 9,278 |
| 0.50 | 0.869 | 0.938 | 0.963 | 0.989 | 0.896 | 2,555 |

Source: `results/buckets_base_p2_1280_val.csv`; the 0.25 row is in
`results/runs.csv`. A further 10,736 val boxes (31%) are under 10 px and are
not scored. The test split (`ka`, `mms`) is still untouched.

**Speed.** On the Jetson Orin Nano Super (MAXN_SUPER, 736x1280 FP16 engine),
the model takes **29.2 ms median / 32.4 ms p95 per frame** (pre + inference +
post), about 34 Hz. With the 30 fps Brio 105 webcam, the live loop is capped
by the camera. On a Mac M4 (MPS, PyTorch) it is 24.2 ms median / 39.8 ms p95
(`results/latency.csv`).

**Live check, indoors (Brio 105, close range).** Cones 1–3 m away are boxed at
conf 0.75–0.90 and stay boxed through motion blur and hand occlusion. No false
positives on shoes, chairs, a backpack or an orange keychain. Weak spots:
stacked or nested cones get one box for the stack plus partial boxes inside it
(conf 0.3–0.6), and in a tight line of cones the ones hidden behind come out
at conf 0.3–0.4. Range against distant cones has not been tested live yet.

## Tasks

| # | Task | Script | Status |
|---|------|--------|--------|
| 0 | Env + readiness check | `scripts/env_check.py` | **done** |
| 1 | Inventory both datasets | `scripts/inventory.py`, `scripts/fsoco_to_yolo.py` | **done** — see `notes/findings.md` |
| 2 | Inspect the AKS `1` class | — | **done — it is real cones, merge it** |
| 3 | Merge: collapse to 1 class, phash dedupe, split by prefix | `scripts/merge.py` | **done** — 13,847 imgs / 227,317 boxes |
| 4 | Baseline: **yolo26n-p2 @ 1280**, close_mosaic 10 | `scripts/train.py` | **done** — 50 epochs, Colab T4 |
| 5 | Eval: recall by bbox-height bucket | `scripts/eval_buckets.py` | **done** — see Results |
| 6 | TensorRT on the Jetson, end-to-end latency, live camera | `scripts/jetson/` | **running** — 29 ms/frame; distant-cone range test pending |
| 7 | Quantify loss at 640 / 960 vs the 1280 baseline | `scripts/train.py --imgsz` | not started |

## Datasets

The images are **not in this repo** (24 GB raw). Put them in place, then
convert and merge:

```
~/fsoco_raw/        # FSOCO bounding boxes, Supervisely format (41 team folders + meta.json)
data/raw/aks/       # AKS cone set, Roboflow YOLOv11 export - 2 classes, both real cones
data/interim/fsoco/ # written by scripts/fsoco_to_yolo.py
data/merged/        # written by scripts/merge.py - never edit by hand
```

```bash
python scripts/fsoco_to_yolo.py ~/fsoco_raw -o data/interim/fsoco
python scripts/merge.py          # also writes configs/cone.yaml for this machine
```

**Overlap is much bigger than expected:** 2,997 AKS images (56.5%) stem-match an
FSOCO image — prefixes `amz`, `bme`, `gfr`, `ulm`, `mms`, `frt`, `ka`, `tuwr`,
`pwrrt` are all FSOCO teams. Roboflow re-encodes and renames
(`bme_00625_jpg.rf.<hash>.jpg`), so phash is still needed to catch the rest.

**And the overlapping copies disagree:** AKS is missing 66,238 boxes relative to
FSOCO on those same images — it labels only the large near cones. See
`notes/findings.md`. The merge must prefer FSOCO labels, not just deduplicate.

**Splitting:** by **lowercased filename prefix** (= source team), never randomly
and never by folder — folder is the *labeling* team, and `amz_*` alone spans 20
folders. Both sets are video-derived; adjacent frames are near-duplicates, so a
random split leaks val into train and reports a recall we do not have.
`amz` is 34% of FSOCO, which constrains how the prefixes can be balanced.
Folder `prom` holds all 98 issue-tagged images → train only, never val/test.

## Setup

```bash
conda create -n cone python=3.11 && conda activate cone
pip install torch torchvision          # CUDA build on a GPU box; default wheel on a Mac
pip install -r requirements.txt
python scripts/eval_buckets.py --self-test   # no data needed
python scripts/env_check.py                  # needs the datasets in place
```

Use Python 3.11: newer interpreters (3.14) have no torch/ultralytics wheels yet.
The author's Mac runs torch 2.14.0 (MPS) and ultralytics 8.4.155; the baseline
was trained on 8.4.177. RF-DETR and DEIMv2 each pin their own torch, so they get
separate envs: `scripts/setup_rfdetr_env.sh`, `scripts/setup_deimv2_env.sh`.

Just running the model on the Jetson needs none of this: see
[`scripts/jetson/README.md`](scripts/jetson/README.md).

## imgsz — decided from the data, not a sweep

Post-letterbox box-height distribution, share clearing the ~10px labeling floor:

| imgsz | usable boxes |
|---|---|
| 640 | 30.6% |
| 960 | 47.1% |
| **1280** | **60.7%** |

→ **Train at imgsz 1280 with the P2 head.** 640/960 are run only to quantify
what they cost, not as candidates. Budget VRAM accordingly — 1280 + P2 is
memory-hungry.

Queued experiments once a baseline exists:
- filter boxes `<6px` out of training (14.2% of labels at 1280)
- `rect=True` — the most common source resolution is 2872x920, a 3.1:1 ultrawide

The ~10px floor is FSOCO's labeling policy ("100 square pixels"), so **recall
below 10px is not measurable** — absence of a label there does not mean absence
of a cone. The `<10px` bucket stays in the eval as a diagnostic, not a target.

## Negative results — do not retry

- **`yolo26n-p2 @ imgsz 1280` does not train on M4 MPS.** A probe on a 411-image
  slice (4% of train, batch 4) did not finish a single epoch in 20+ minutes.
  Extrapolated, one full epoch is measured in hours and the 100-epoch baseline
  is weeks. The P2 head adds a stride-4 feature map — at 1280 that is a
  320×320 grid, and MPS has no efficient path for it.
  → **Training happens on a cluster GPU.** `scripts/train.py` takes `--device`
  and `--batch` so the same script runs both places; the Mac is for data prep,
  eval and export only.

## Gotchas already paid for

- **`path:` in a dataset yaml must be absolute.** Ultralytics resolves a
  relative `path:` against the *current working directory* (then against
  `datasets_dir`), **not** against the yaml's own location. A relative path
  silently points somewhere else depending on where you launch from.
- **`runs_dir` was pointing at `/opt/homebrew/runs`.** Now set to
  `runs/` inside the project, so runs stay with the code they came from.
  Same for `datasets_dir` and `weights_dir` (`yolo settings`).
- **AKS `names: ['1', 'cone']` — id 0 is `'1'`, id 1 is `cone`.** Reversed from
  intuition; assuming `0 == cone` inverts the dataset.
- **TensorRT export cannot happen on this Mac.** Task 6 builds the engine
  *on the Jetson*; the Mac side only gets as far as ONNX. Engines are not
  portable across devices or TensorRT versions anyway.

## Training

`scripts/train.py` hardcodes nothing about the machine — same script locally and
on a cluster.

```bash
# see the resolved config without training
python scripts/train.py --dry-run

# the baseline as it was run (Colab T4, 50 epochs)
python scripts/train.py --weights yolo26n.pt --device 0 --batch 8 --workers 2 \
    --epochs 50 --name base_p2_1280

# after a disconnect or walltime kill: the same command plus --resume
python scripts/train.py ... --resume

# quick plumbing check anywhere
python scripts/train.py --model yolo26n.pt --imgsz 640 --batch 8 \
    --epochs 1 --fraction 0.02 --name smoke
```

**Always pass `--weights yolo26n.pt`.** The default `--model` is a `.yaml`,
which is only an architecture: without `--weights` the network starts from
random weights. With it, the COCO weights load into every layer that fits
(the P2 head's extra layers start fresh).

`--resume` loads `<project>/<name>/weights/last.pt`. Ultralytics only resumes
from a checkpoint; `--resume` on a `.yaml` silently restarts at epoch 0.

Defaults are the decided policy: `yolo26n-p2.yaml`, `imgsz 1280`,
`close_mosaic 10`, `fliplr 0.5`, `hsv_h 0.0`. `--device auto` resolves
cuda > mps > cpu. On a T4, AutoBatch (`--batch -1`) picks 2, which is too
small; batch 8 fits in about 10 GB. Runs land in `yolo26n/runs/<name>/`.
The script refuses to start a P2 run at imgsz≥1280 on MPS rather than let you
discover the negative result again.

Queued experiments have flags already: `--min-box-px` (in `merge.py`) and
`--rect`.

### Training on Colab

A free T4 does about 14 min per epoch at 1280 (CPU-bound on 2 cores), so
50 epochs is about 11 h across 2–3 sessions. Pack the merged set once with
`COPYFILE_DISABLE=1 tar --no-xattrs -cf cone_merged.tar -C data merged`
(otherwise macOS adds a `._*` junk file next to every image), upload it and a
zip of the repo to Drive, then:

```python
from google.colab import drive; drive.mount('/content/drive')
D = '/content/drive/MyDrive/aks-cone'
!mkdir -p /content/data && tar -xf {D}/cone_merged.tar -C /content/data 2>/dev/null; find /content/data/merged -name '._*' -delete
!cd /content && unzip -qo {D}/cone_code.zip -d aks && pip -q install ultralytics
open('/content/aks/configs/cone.colab.yaml','w').write(
"path: /content/data/merged\ntrain: train/images\nval: val/images\ntest: test/images\nnc: 1\nnames:\n  0: cone\n")
!cd /content/aks && python scripts/train.py --data configs/cone.colab.yaml --weights yolo26n.pt \
   --device 0 --batch 8 --workers 2 --epochs 50 --name base_p2_1280 --project {D}/runs
```

Extract to `/content`, not Drive: reading thousands of small files from Drive
is slow. Checkpoints go to Drive (`--project`), so after a disconnect, rerun
the cells with `--resume` added.

SLURM jobs for Purdue RCAC clusters are in `scripts/slurm/`.

## Eval

`scripts/eval_buckets.py` — recall per size bucket, which is the number that
decides things. Overall mAP is logged for context and otherwise ignored.

```bash
python scripts/eval_buckets.py --weights models/base_p2_1280.pt --run base_p2_1280
python scripts/eval_buckets.py --weights ... --conf 0.05,0.1,0.25,0.5 --split test
python scripts/eval_buckets.py --self-test        # no model, no data needed
```

**Buckets: `10-16` · `16-32` · `32-64` · `>=64` px**, measured in
**post-letterbox pixels at the eval imgsz** — source resolutions run 2872x920
to 6304x4304, so a source-pixel bucket means nothing.

**Buckets floor at 10px.** FSOCO's labeling policy is ~100 square pixels, so
below ~10px a missing label does not mean a missing cone. Sub-floor boxes are
counted and reported but **not scored** — and a prediction landing on one is
neither credited nor penalised, since that region is effectively unlabelled.

How much of the val set is even scorable depends entirely on imgsz:

| imgsz | scorable | sub-floor (unscorable) |
|---|---|---|
| 640 | 34.7% | 65.3% |
| 960 | 52.9% | 47.1% |
| **1280** | **68.9%** | 31.1% |

`--conf` sweeps confidence so the recall/confidence trade-off is visible per
bucket; `--log-conf` picks which one is written to `results/runs.csv`. Full
sweep goes to `results/buckets_<run>_<split>.csv`.

## Run log

Every run appends config + bucketed recall to `results/runs.csv`
(`scripts/runlog.py`), starting from run one. Per-dataset inventories land in
`results/inventory_<name>.json`.

## Training defaults

- `model=yolo26n-p2.yaml`, `imgsz=1280` — see the imgsz section above
- start from `--weights yolo26n.pt` (COCO-pretrained)
- keep `fliplr` — a cone is symmetric, the flip is free data
- `hsv_h` ≈ 0 — cone hue is the signal; do not rotate it away
- `close_mosaic 10`
- recall over precision; no high confidence threshold — false positives are
  filtered temporally downstream
