# CLAUDE.md — cone obstacle detection (Autonomous Karting Series)

Handoff context. Detail lives in `README.md` (decisions + usage) and
`notes/findings.md` (evidence for every claim here).

## What this is

Detect traffic cones as **mid-track obstacles to steer around**. The track is
unbounded (road bleeds into grass), so cones are **not** boundary markers and
color carries no meaning. Single class `cone`. Target hardware: Jetson,
TensorRT.

**We optimize recall bucketed by bbox height, not mAP.** A missed cone is a
collision; false positives get filtered temporally downstream, so we do *not*
buy precision with a high confidence threshold. Overall mAP is dominated by big
near cones that were never at risk of being missed — it is logged for context
and otherwise ignored.

## Environment

```bash
conda activate cone          # python 3.11.16, torch 2.14.0 (MPS), ultralytics 8.4.155
python scripts/env_check.py  # 11 checks: env, weights, config, both datasets, merged splits
```

- The base miniforge env is **Python 3.14 — no torch/ultralytics wheels**. Never use it.
- Env interpreter: `/opt/homebrew/Caskroom/miniforge/base/envs/cone/bin/python`
- Machine is an Apple M4 (MPS). `yolo26n` loads (2.57M params); `yolo26n-p2.yaml`
  builds (2.66M). Both pretrained `.pt` cached in `weights/`.
- ultralytics `runs_dir`/`datasets_dir`/`weights_dir` are pointed **into this
  project** (via `yolo settings`), not `/opt/homebrew`.

## Where the data is

| path | what | size |
|---|---|---|
| `~/fsoco_raw/` | FSOCO Supervisely, 41 team folders + `meta.json` | 24 GB |
| `~/fsoco_boxes.zip` | the original download | 23 GB |
| `data/interim/fsoco/` | converted to YOLO + `manifest.csv` | 5.5 GB |
| `data/raw/aks/` | AKS Roboflow YOLOv11 export | 1.3 GB |
| `data/merged/` | **the training set** — hardlinked, so ~no extra disk | 56 MB |

A third FSOCO set (kart images only) was **deliberately not downloaded** — out
of scope for now.

## Status

| # | task | state |
|---|---|---|
| 0 | env + readiness | **done** — `scripts/env_check.py`, 11/11 green |
| 1 | inventory both sets | **done** — `notes/findings.md` |
| 2 | inspect AKS class `1` | **done** — it is real cones, merged |
| 3 | merge / dedupe / split | **done** — `data/merged`, 13,847 imgs / 227,317 boxes |
| 4 | baseline train | **done** — 50 epochs on Colab T4 from `yolo26n.pt`; `models/base_p2_1280.pt` |
| 5 | bucketed-recall eval | **done** — val recall @0.25: 0.934 / 0.967 / 0.981 / 0.993, precision 0.711 |
| 6 | TensorRT export + e2e Jetson latency | **running** — `scripts/jetson/`; Orin Nano Super 29.2 ms median / 32.4 p95; distant-cone range test pending |
| 7 | quantify 640/960 loss vs 1280 | not started |

Code lives at github.com/nyaniv/aks-cone-detection (public). Datasets, runs and
engines are gitignored; `configs/cone.yaml` is generated per machine by `merge.py`.

## Decisions already made (do not relitigate)

- **imgsz 1280 + P2 head is the baseline**, decided from the data, not a sweep:
  share of boxes clearing the 10px labeling floor after letterbox is
  30.6% @640, 47.1% @960, **60.7% @1280**. 640/960 get run only to quantify
  what they cost (task 7).
- **Split by lowercased filename prefix** (= source team), never randomly and
  never by folder — folder is the *labeling* team and `amz_*` spans 20 folders.
  Both sets are video-derived, so a random split leaks val into train.
  - val = `bme`,`ff` (1,775) · test = `ka`,`mms` (1,776) · everything else train
  - `amz` (34% of FSOCO) and `prom` (all 98 issue-tagged images) forced to train
- **val/test are FSOCO-only.** AKS labels are incomplete (see below), so recall
  measured against them would measure nothing.
- **Native-AKS (2,304 imgs) is train-only** — our only karting-domain data, but
  its labels stop at ~25px. User decision, 2026-09-18.
- **AKS class `1` is real cones, merged into `cone`** — not junk.
- Keep `fliplr` (a cone is symmetric); `hsv_h` ≈ 0 (cone hue is the signal).
- Queued experiments, flags already exist: `merge.py --min-box-px 6`,
  `train.py --rect`.

## The findings that shaped the above

- **AKS labels only large near cones.** On the 2,997 images AKS shares with
  FSOCO, AKS is missing **66,238 boxes**; 2,913 images have fewer, zero have
  more (`amz_01945`: 4 vs 64). Median box height 67px vs FSOCO's 25px. Mixing
  these naively gives contradictory supervision — FSOCO says a 12px blob is a
  cone, AKS says it is background — which trains our actual objective away.
- **AKS `names: ['1','cone']` → id 0 is `'1'`, id 1 is `cone`.** Reversed from
  intuition. Assuming `0 == cone` inverts the dataset.
- **FSOCO labeling floor is ~100 sq px (~10x10).** Below ~10px, a missing label
  does not mean a missing cone, so recall there is not measurable.
- 11,572 FSOCO images, **220,862 boxes**, all tagged `train` (no `test` tag
  exists). 98 images carry 114 issue tags, all in folder `prom`.

## Traps that already cost time — do not rediscover

- **`yolo26n-p2 @ 1280 does not train on M4 MPS.** A 411-image probe did not
  finish one epoch in 20+ min. `train.py` now refuses this combination.
  Training happens on a cluster GPU; the Mac is for data prep, eval, export.
- **A dataset yaml's `path:` must be absolute.** Ultralytics resolves a relative
  `path:` against the **CWD** (then `datasets_dir`), *not* the yaml's location.
  `merge.py` writes `configs/cone.yaml` with an absolute path; `env_check.py`
  asserts it.
- **64-bit phash cannot separate adjacent video frames.** The first merge
  dropped 622 FSOCO images as "duplicates"; sampling 12 showed *none* were —
  consecutive frames with different labels (one pair 3 boxes vs 9). The
  internal pass now confirms a phash match with mean pixel difference; only
  **29** survived. Do not loosen this back to phash-only.
- **`prefix_of` requires a `_`/`-` separator.** Without it, hex-named AKS files
  (`ff1e2b80…`) yield a junk prefix `ff` that collides with the real FSOCO team
  `ff_*` and looks exactly like split leakage. AKS prefixes are namespaced
  `aks:*`.
- **TensorRT export cannot happen on this Mac** — task 6 builds the engine *on
  the Jetson*; the Mac only reaches ONNX. Engines are not portable across
  devices or TRT versions anyway.
- Don't evaluate against a COCO-pretrained model to "test" the eval script — no
  cone class, detects nothing, proves nothing. Use `--self-test`.

## Scripts

| script | what |
|---|---|
| `env_check.py` | 11-point readiness check; run it first after any env change |
| `imgmeta.py` | stdlib-only image-header dims, so task 1 never blocks on deps |
| `inventory.py` | counts, class names, dims, bbox-height histogram (stdlib only) |
| `fsoco_to_yolo.py` | Supervisely → YOLO, rectangles only, downscale, writes `manifest.csv` |
| `merge.py` | collapse to 1 class, dedupe, split by prefix, rewrites `configs/cone.yaml` |
| `train.py` | `--data/--model/--imgsz/--batch/--device`; runs locally and on a cluster |
| `eval_buckets.py` | recall per size bucket, `--conf` sweep, `--self-test` |
| `runlog.py` | appends every run's config + bucketed recall to `results/runs.csv` |
| `backends.py` | the only model-specific code in the eval path; contract is boxes in SOURCE px |
| `make_subset.py` | ONE sampled image list, materialised as YOLO *and* COCO |
| `yolo_to_coco.py` | `data/merged` -> COCO mirror, shared by rf-detr and DEIMv2; mirrors the split, never re-splits |
| `train_rfdetr.py` | RF-DETR on the same splits (env `rfdetr`) |
| `train_deimv2.py` | DEIMv2 on the same splits (env `deimv2`); writes the run's `config.yml` |
| `deimv2_cfg.py` | builds that config by overriding an upstream DEIMv2 one |
| `predict.py` | GT + predictions drawn for any backend, same frames per `--seed` |
| `compare.py` | matched-protocol yolo26n vs rf-detr |
| `jetson/` | on the Jetson: `run_container.sh` (host), `export_engine.sh` (TRT FP16 736x1280), `camera_loop.py` (threaded Brio capture, `--save` video) |

Three conda envs, on purpose - each model pins its own torch. `cone` is the one
`env_check.py` validates; `rfdetr` and `deimv2` are set up by
`scripts/setup_*_env.sh`. Per-model notes, including the traps specific to each,
live in **`rfdetr/README.md`** and **`deimv2/README.md`**.

```bash
python scripts/train.py --dry-run                                   # resolved config, no run
python scripts/train.py --device 0 --batch -1 --name base_p2_1280   # the baseline, on a GPU
python scripts/eval_buckets.py --weights runs/base_p2_1280/weights/best.pt
python scripts/eval_buckets.py --self-test
```

## Eval buckets

`10-16 / 16-32 / 32-64 / >=64` px, measured in **post-letterbox pixels at the
eval imgsz** — source resolutions run 2872x920 to 6304x4304, so source-pixel
buckets are meaningless. Buckets **floor at 10px**; sub-floor boxes are reported
but not scored, and a prediction landing on one is neither credited nor
penalised.

Share of the val set that is even scorable: **34.7% @640 · 52.9% @960 ·
68.9% @1280.**

## Next step

1. Range test on the Jetson: cones at 5 / 10 / 15 / 20 m outdoors, with
   `scripts/jetson/camera_loop.py --save`. Everything tested live so far was
   1-3 m away, which says nothing about the small-cone objective.
2. Confirm the live loop reaches ~30 Hz with the threaded grabber. Every saved
   video shows 15 Hz; the timing summary prints the camera's delivered fps,
   which tells camera auto-exposure apart from a slow loop.
3. Task 7: retrain at `--imgsz 960` and `640` to measure what they cost.
4. Test split (`ka`, `mms`) only for the final model comparison.
