# RF-DETR vs yolo26n

Same data, same splits, same metric. The only thing that varies is the model.

## Why the comparison is valid

- **Same splits.** `yolo_to_coco.py` mirrors `merge.py`'s split; it never
  re-splits. Verified: 158,128 / 34,488 / 34,701 boxes, identical to the YOLO
  source, 227,317 total.
- **Same metric.** Both go through `scripts/eval_buckets.py`, whose `score()`
  is pure and covered by 9 self-tests. Only `backends.py` differs, and its
  contract is *boxes in SOURCE pixels* — each backend undoes its own
  letterbox/resize before returning.
- **Same ledger.** Both append to `results/runs.csv`. That file is the
  comparison table.

## Setup (separate env, on purpose)

```bash
conda create -y -n rfdetr python=3.11 && conda activate rfdetr
pip install -r requirements-rfdetr.txt   # rfdetr[train,loggers] - the bare
                                        # package is INFERENCE ONLY
python scripts/yolo_to_coco.py          # once; hardlinks, ~no extra disk
```

## Resolution — per variant, NOT a constant

RF-DETR needs a resolution divisible by `patch_size * num_windows`, and that
product differs by variant (verified against rfdetr 1.10.1 source):

| variant | patch x windows | step | 1280 valid? |
|---|---|---|---|
| Nano / Small / Medium / Large | 16 x 2 | **32** | **yes** |
| Base | 14 x 4 | 56 | no (use 1344) |

So with Nano or Small — the variants you actually want for Jetson — **1280 is
already byte-identical to the yolo26n baseline**. Only `RFDETRBase` forces
LCM(32, 56) = 224, i.e. 1120 or 1344.

`backends.resolution_step()` reads this from the variant's config rather than
hardcoding, and `compare.py` asks the rfdetr interpreter for it (rfdetr is not
importable from the `cone` env).

## Run

```bash
python scripts/train_rfdetr.py --dry-run
python scripts/train_rfdetr.py --arch RFDETRSmall --resolution 1280 \
    --epochs 100 --name rfdetr_s_1280

python scripts/eval_buckets.py --backend rfdetr --arch RFDETRSmall \
    --weights rfdetr/runs/rfdetr_s_1280/checkpoint_best_total.pth \
    --imgsz 1280 --run rfdetr_s_1280

python scripts/predict.py --backend rfdetr --arch RFDETRSmall \
    --weights rfdetr/runs/rfdetr_s_1280/checkpoint_best_total.pth \
    --imgsz 1280 --n 8 --conf 0.10 --topk 25
```

`--seed 0` (the default) picks the **same 8 images** for both backends, so
`yolo26n/viz/` and `rfdetr/viz/` are comparable frame by frame.

## Which arch to compare

yolo26n is 2.5M params and targets Jetson/TensorRT. **RFDETRNano or
RFDETRSmall** is the fair size match; Base/Large would win on accuracy while
losing the deployment argument. Consider running both and reporting the
latency/recall trade-off, since task 6 is an e2e Jetson latency budget.

## Differences that will show up in the numbers

- **No NMS.** RF-DETR is a set predictor — every returned box is a distinct
  query. There is no duplicate suppression to tune, so its FP column behaves
  differently from YOLO's at low confidence.
- **Query budget** is ~300. This dataset peaks at 122 boxes in one image
  (p99 = 75), so there is headroom — but if you ever loosen `--min-box-px`,
  re-check it.
- **Small objects** are DETR's classic weakness and *your entire objective*.
  The 10-16px bucket is the number that decides this comparison, not mAP.


---

# Training on Purdue RCAC

| file | what |
|---|---|
| `scripts/slurm/setup_rfdetr_rcac.sh` | one-time env on the cluster (CUDA torch, then rfdetr[train]) |
| `scripts/slurm/train_rfdetr.sub` | the sbatch job; every knob is an env var |
| `scripts/slurm/sync_to_rcac.sh` | code + `data/merged` to `$RCAC_SCRATCH` |

## 0. Fill in two values

`train_rfdetr.sub` ships with `--account=CHANGE_ME` and `--partition=CHANGE_ME`.
`slist` prints what you can charge; `standby` is free but caps walltime at 4 h,
which is what `CHAIN=1` exists for.

## 1. Push code + data

```bash
./scripts/slurm/sync_to_rcac.sh you@gilbreth.rcac.purdue.edu
```

~5.7 GB. The COCO mirror is **not** transferred — it is hardlinked to
`data/merged` locally, so sending both would move the same bytes twice.

## 2. Env + COCO mirror, on the cluster

```bash
sinteractive -A <account> -p <queue> --gres=gpu:1 -t 00:30:00
bash scripts/slurm/setup_rfdetr_rcac.sh      # installs CUDA torch, then rfdetr[train,loggers]
python scripts/yolo_to_coco.py               # rebuild the mirror there (hardlinks, ~free)
```

The setup script prints `nvidia-smi` first — if the driver is older than cu124
expects, rerun with `CUDA_CHANNEL=cu121`.

## 3. Submit

```bash
sbatch scripts/slurm/train_rfdetr.sub                     # RFDETRSmall @1280, 100 epochs
squeue -u $USER
tail -f logs/cone-rfdetr-<jobid>.out
```

Smoke test first (finishes in minutes, confirms env + data + GPU):

```bash
sbatch --export=ALL,ARCH=RFDETRNano,RESOLUTION=640,EPOCHS=1,RUN_NAME=rf_smoke,CHAIN=0 \
  scripts/slurm/train_rfdetr.sub
```

Multi-GPU on one node — raise `--gres=gpu:4`, then:

```bash
sbatch --export=ALL,DEVICES=4,STRATEGY=ddp,BATCH=4,GRAD_ACCUM=1 \
  scripts/slurm/train_rfdetr.sub
```

## Matching the yolo26n run

For a comparison that means anything, both models need the same splits,
resolution and schedule:

```bash
sbatch --export=ALL,RUN_NAME=base_p2_1280                      scripts/slurm/train.sub
sbatch --export=ALL,ARCH=RFDETRSmall,RESOLUTION=1280,RUN_NAME=rf_s_1280 \
  scripts/slurm/train_rfdetr.sub
```

1280 is valid for **both** (yolo %32, RFDETRSmall %32). Only `RFDETRBase` would
force 1344. Both jobs append to the same `results/runs.csv`, then:

```bash
python scripts/compare.py --stage report --imgsz 1280 --rfdetr-arch RFDETRSmall --tag base
```

## What the job handles

- **Walltime overruns.** Training is wrapped in `timeout` with a 15 min margin
  (rf-detr checkpoints are ~500 MB), so the tail always runs; it resubmits
  itself and `--resume last.ckpt` picks up. `CHAIN_MAX=12` bounds it.
- **`checkpoint_interval=1`** so a kill loses at most one epoch.
- **Caches off `$HOME`** (25 GB quota): `TORCH_HOME`, `HF_HOME`, `MPLCONFIGDIR`,
  and `ROBOFLOW_HOME` — rf-detr downloads its pretrained backbone to the last one.
- **`--mem=96G`**, higher than the yolo job: 30.5M params plus transformer
  activations at 1280.
- **Node-local staging** to `$TMPDIR` (`STAGE_LOCAL=0` to read from scratch).
- **Eval on clean finish** — bucketed recall into `results/runs.csv`.
