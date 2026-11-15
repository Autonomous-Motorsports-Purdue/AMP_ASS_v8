# DEIMv2 vs yolo26n (and rf-detr)

Same data, same splits, same metric. The only thing that varies is the model.

## Why the comparison is valid

- **Same splits.** DEIMv2 eats the *same* `data/merged_coco` that rf-detr does,
  built by `yolo_to_coco.py`, which mirrors `merge.py`'s split and never
  re-splits. 158,128 / 34,488 / 34,701 boxes, identical to the YOLO source.
- **Same metric.** All three go through `scripts/eval_buckets.py`, whose
  `score()` is pure and covered by 9 self-tests. Only `backends.py` differs,
  and its contract is *boxes in SOURCE pixels* — each backend undoes its own
  resize before returning.
- **Same ledger.** Everything appends to `results/runs.csv`. That file is the
  comparison table.

## Setup (a third env, on purpose)

DEIMv2 is a **repo, not a pip package** — the clone *is* the install, and it
pins `torch==2.5.1`, so it cannot share `cone` (torch 2.14.0, what
`env_check.py` validates) or `rfdetr` (its own pin).

```bash
bash scripts/setup_deimv2_env.sh      # clones DEIMv2, creates env `deimv2`
python scripts/yolo_to_coco.py        # only if rf-detr has not built it already
python scripts/train_deimv2.py --dry-run
```

The clone lands in `deimv2/DEIMv2/` and is **gitignored** — it is an install.
`$DEIMV2_HOME` overrides the location, which is how the cluster keeps it on
scratch.

## Resolution — one rule, unlike rf-detr

Every DEIMv2 backbone is satisfied by a **multiple of 32**: HGNetv2 downsamples
32x, and the ViT-Tiny/DINOv3 backbones use patch 16. There is no per-variant
window product to look up the way `RFDETRBase`'s 56 has to be.

So **1280 is valid for every DEIMv2 variant** — byte-identical to the yolo26n
baseline and to RFDETRNano/Small. `deimv2_cfg.RESOLUTION_STEP` is where that
lives if a future variant ever disagrees.

## The trap that will cost you a run: `num_classes` is 2, not 1

DEIMv2 with `remap_mscoco_category: False` uses **`category_id` itself as the
training label**. Our COCO mirror follows Roboflow's convention — id 0 is a
dummy supercategory `cones`, id 1 is the real class `cone` — so the label is
**1**, and the head needs **`num_classes: 2`**. Setting 1 builds a head that
cannot represent the only class in the dataset.

`deimv2_cfg.coco_num_classes()` reads this off `_annotations.coco.json`
(`max(category_id) + 1`) rather than trusting anyone to remember it.

## The difference that will show up in the numbers: DEIMv2 **stretches**

yolo26n and rf-detr letterbox. DEIMv2 resizes to a **square** and does not
preserve aspect ratio. On this dataset that is not a detail — source images run
from 2872x920 to 6304x4304.

For a 2872x920 frame at 1280, a cone 20px tall in the source becomes:

| | vertical scale | 20px cone becomes |
|---|---|---|
| letterbox (yolo26n, rf-detr) | 1280/2872 = 0.446 | **8.9 px** |
| square stretch (DEIMv2) | 1280/920 = 1.391 | **27.8 px** |

Read that carefully before reading the results table:

- **The metric is unaffected.** `eval_buckets.py` buckets ground truth by
  *post-letterbox* height computed from the source box — a property of the
  measuring instrument, not of the backend. All three models are still scored
  on the identical set of boxes in the identical buckets.
- **What the model sees is very much affected.** On wide frames DEIMv2 gets a
  3x taller view of exactly the small cones that decide this project, and pays
  for it with distorted aspect ratio and a squeezed horizontal axis.

That is a real preprocessing advantage/disadvantage, not a bug and not
something to normalise away silently. Report it next to the numbers. If you
ever want the letterboxed variant for a controlled comparison, it is a new
transform in the generated `ops` list, not a flag.

## Which variant to compare

yolo26n is 2.57M params and targets Jetson/TensorRT. The HGNetv2 family is the
fair size match; the DINOv3 family wins on AP while losing the deployment
argument.

| variant | `--arch` | #Params | GFLOPs | COCO AP | backbone auto-downloads? |
|---|---|---|---|---|---|
| Atto | `hgnetv2_atto` | 0.5M | 0.8 | 23.8 | yes |
| Femto | `hgnetv2_femto` | 1.0M | 1.7 | 31.0 | yes |
| **Pico** | `hgnetv2_pico` | **1.5M** | 5.2 | 38.5 | yes |
| **N** | `hgnetv2_n` | **3.6M** | 6.8 | 43.0 | yes |
| S | `dinov3_s` | 9.7M | 25.6 | 50.9 | **no** — `ckpts/vitt_distill.pt` |
| M | `dinov3_m` | 18.1M | 52.2 | 53.0 | **no** — `ckpts/vittplus_distill.pt` |
| L | `dinov3_l` | 32.2M | 96.7 | 56.0 | no — DINOv3-S, see upstream README |
| X | `dinov3_x` | 50.3M | 151.6 | 57.8 | no — DINOv3-S+, see upstream README |

`hgnetv2_pico` (1.5M) and `hgnetv2_n` (3.6M) bracket yolo26n's 2.57M. Start
with `hgnetv2_n`.

**`--arch s` is rejected on purpose**: DEIMv2 ships *both* `hgnetv2_s` and
`dinov3_s`, which are different models. Spell the backbone.

### Backbone checkpoints

`hgnetv2_*` fetches its own backbone. `dinov3_s` / `dinov3_m` need
`vitt_distill.pt` / `vittplus_distill.pt` dropped into `$DEIMV2_HOME/ckpts/`
(links in the upstream README's model zoo); `train_deimv2.py` checks for them
and stops before wasting a GPU allocation.

## Run

```bash
python scripts/train_deimv2.py --dry-run                 # resolved config, no run
python scripts/train_deimv2.py --arch hgnetv2_n --resolution 1280 \
    --name deimv2_n_1280 --device cuda

python scripts/eval_buckets.py --backend deimv2 --arch hgnetv2_n \
    --weights deimv2/runs/deimv2_n_1280/best_stg2.pth \
    --imgsz 1280 --run deimv2_n_1280

python scripts/predict.py --backend deimv2 --arch hgnetv2_n \
    --weights deimv2/runs/deimv2_n_1280/best_stg2.pth \
    --imgsz 1280 --n 8 --conf 0.10 --topk 25
```

`--seed 0` (the default) picks the **same 8 images** for every backend, so
`yolo26n/viz/`, `rfdetr/viz/` and `deimv2/viz/` are comparable frame by frame.

### The run carries its own config

`train_deimv2.py` writes **`config.yml` into the run directory** and points
DEIMv2's `train.py` at it. That file `__include__`s the upstream variant config
by absolute path and overrides only dataset, classes, resolution and schedule —
so upgrading DEIMv2 is a `git pull`, not a merge.

`eval_buckets.py --backend deimv2` reads that same `config.yml` back from next
to the checkpoint. It has to: the checkpoint only loads against the config that
built it. `--model-config` overrides the location.

### `--epochs` rescales the schedule, it does not truncate it

Upstream tunes `flat_epoch`, `no_aug_epoch`, the transform `policy.epoch` list
and the collate `stop_epoch` / `mixup_epochs` / `copyblend_epochs` against its
own `epoches`. Asking for 20 epochs while those still say 29/50 means the
no-aug tail and the matcher change **never happen**. So `--epochs N` scales all
of them by the same factor and prints the mapping. `--epochs 0` (the default)
keeps the variant's own tuned schedule untouched.

### Fine-tuning from COCO

```bash
python scripts/train_deimv2.py --arch hgnetv2_n --tuning /path/deimv2_n_coco.pth
```

`-t` upstream. The class head will not match 2 classes and is skipped — what
you are reusing is the backbone and encoder.

## Differences that will show up in the numbers

- **No NMS.** Like rf-detr, DEIMv2 is a set predictor — every returned box is a
  distinct query. There is no duplicate suppression to tune, so its FP column
  behaves differently from YOLO's at low confidence.
- **Query budget is 300.** This dataset peaks at 122 boxes in one image
  (p99 = 75), so there is headroom — but if you ever loosen `--min-box-px`,
  re-check it.
- **Heavy augmentation by default**: Mosaic, RandomIoUCrop, RandomZoomOut,
  mixup and copy-blend, staged by epoch. `copyblend`'s `area_threshold: 100`
  happens to sit exactly on FSOCO's ~100 sq px labeling floor.
- **`RandomPhotometricDistort` randomises hue**, which our `hsv_h ~ 0` decision
  deliberately avoids — cone hue is signal. `--no-photometric` drops it (from
  the ops list *and* the aug policy). Worth running both ways.
- **Small objects** are the classic transformer weakness and *your entire
  objective*. The 10-16px bucket decides this comparison, not mAP.

## Three-way comparison

`scripts/compare.py` is a matched-protocol harness for yolo26n vs rf-detr only.
Until it grows a third leg, run DEIMv2 through the same knobs by hand — the
resolution, splits, `--conf` sweep, `--limit` and `--seed` are what have to
match — and read all three rows out of `results/runs.csv`.


---

# Training on Purdue RCAC

| file | what |
|---|---|
| `scripts/slurm/setup_deimv2_rcac.sh` | one-time env + clone on the cluster (CUDA torch at DEIMv2's pin, then the rest) |
| `scripts/slurm/train_deimv2.sub` | the sbatch job; every knob is an env var |
| `scripts/slurm/sync_to_rcac.sh` | code + `data/merged` to `$RCAC_SCRATCH` (skips the clone) |

## 0. Fill in two values

`train_deimv2.sub` ships with `--account=CHANGE_ME` and `--partition=CHANGE_ME`.
`slist` prints what you can charge; `standby` is free but caps walltime at 4 h,
which is what `CHAIN=1` exists for.

## 1. Push code + data

```bash
./scripts/slurm/sync_to_rcac.sh you@gilbreth.rcac.purdue.edu
```

The DEIMv2 clone is **not** transferred — the cluster clones it itself, onto
scratch, so the CUDA build and the backbone downloads stay off the 25 GB home
quota. The COCO mirror is not transferred either; it is hardlinked to
`data/merged` locally, so sending both would move the same bytes twice.

## 2. Env + clone + COCO mirror, on the cluster

```bash
sinteractive -A <account> -p <queue> --gres=gpu:1 -t 00:30:00
bash scripts/slurm/setup_deimv2_rcac.sh      # clones to $RCAC_SCRATCH/DEIMv2
export DEIMV2_HOME=$RCAC_SCRATCH/DEIMv2      # the job sets this too
python scripts/yolo_to_coco.py               # skip if rf-detr already built it
```

The setup script prints `nvidia-smi` first — if the driver is older than cu124
expects, rerun with `CUDA_CHANNEL=cu121`. It installs torch **before** the rest
of upstream's requirements so nothing pulls a CPU wheel over it.

## 3. Submit

```bash
sbatch scripts/slurm/train_deimv2.sub                     # hgnetv2_n @1280
squeue -u $USER
tail -f logs/cone-deimv2-<jobid>.out
```

Smoke test first (finishes in minutes, confirms env + clone + data + GPU):

```bash
sbatch --export=ALL,ARCH=hgnetv2_pico,RESOLUTION=640,EPOCHS=1,RUN_NAME=dv2_smoke,CHAIN=0 \
  scripts/slurm/train_deimv2.sub
```

Multi-GPU on one node — raise `--gres=gpu:4`, then:

```bash
sbatch --export=ALL,DEVICES=4,BATCH=16 scripts/slurm/train_deimv2.sub
```

`BATCH` is the **total** batch across GPUs (upstream calls it
`total_batch_size`), not per-device. `DEVICES>1` switches the launcher to
`torchrun`.

## Matching the other two runs

```bash
sbatch --export=ALL,RUN_NAME=base_p2_1280                       scripts/slurm/train.sub
sbatch --export=ALL,ARCH=RFDETRSmall,RESOLUTION=1280,RUN_NAME=rf_s_1280 \
  scripts/slurm/train_rfdetr.sub
sbatch --export=ALL,ARCH=hgnetv2_n,RESOLUTION=1280,RUN_NAME=dv2_n_1280 \
  scripts/slurm/train_deimv2.sub
```

1280 is valid for all three (yolo %32, RFDETRSmall %32, every DEIMv2 %32). All
three append to the same `results/runs.csv`.

## What the job handles

- **Walltime overruns.** Training is wrapped in `timeout` with a 15 min margin,
  so the tail always runs; it resubmits itself and `--resume last.pth` picks up.
  `CHAIN_MAX=12` bounds it.
- **Caches off `$HOME`** (25 GB quota): `TORCH_HOME`, `HF_HOME`, `MPLCONFIGDIR`,
  and the clone itself — HGNetv2 backbones and any `ckpts/` downloads land
  beside it on scratch.
- **Node-local staging** to `$TMPDIR` (`STAGE_LOCAL=0` to read from scratch).
- **Eval on clean finish** — bucketed recall into `results/runs.csv`, against
  the first of `best_stg2.pth`, `best_stg1.pth`, `best.pth`, `last.pth` that
  exists.

## Not started

TensorRT export (task 6) for DEIMv2 goes through
`$DEIMV2_HOME/tools/deployment/export_onnx.py`, then `trtexec` **on the
Jetson** — same rule as yolo26n: the Mac only reaches ONNX, and engines are not
portable across devices or TRT versions.
