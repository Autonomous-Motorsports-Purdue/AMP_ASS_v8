# Running task 4 on Purdue RCAC

The Mac cannot train this (`yolo26n-p2 @ 1280` does not finish an epoch on
MPS — see the negative results in the root README). These three files move the
baseline onto a cluster GPU.

| file | what |
|---|---|
| `train.sub` | the sbatch script; every knob is an env var with a default |
| `sync_to_rcac.sh` | rsync code + `data/merged` to `$RCAC_SCRATCH` |
| `README.md` | this — one-time setup |

The other two model tracks mirror these files and are documented with the model,
not here: `train_rfdetr.sub` + `setup_rfdetr_rcac.sh` (**`rfdetr/README.md`**),
`train_deimv2.sub` + `setup_deimv2_rcac.sh` (**`deimv2/README.md`**). All three
jobs use the same walltime/chaining/staging policy on purpose.

## 0. Fill in two values

`train.sub` ships with `--account=CHANGE_ME` and `--partition=CHANGE_ME`.
On the cluster, `slist` prints the accounts you can charge and the queues each
one opens. Gilbreth's `standby` is free but caps walltime at 4 h — that is the
case `CHAIN=1` exists for. `sfeatures` lists GPU types if you want to pin one
with `--constraint`.

## 1. One-time env on the cluster

```bash
module load conda
conda create -y -n cone python=3.11
conda activate cone
# Match the CUDA build to the driver `nvidia-smi` reports on a GPU node.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

`requirements.txt` pins nothing but `ultralytics>=8.3.0`; the Mac env runs
torch 2.14.0 / ultralytics 8.4.155 if you want to match it.

## 2. Push code + data

```bash
./scripts/slurm/sync_to_rcac.sh you@gilbreth.rcac.purdue.edu
```

~5.7 GB — `data/merged` is hardlinked locally, so it is far larger over the
wire than `du` suggests. The transfer is resumable; rerun on a drop.

## 3. Submit

```bash
cd $RCAC_SCRATCH/aks-cone-detection
sbatch scripts/slurm/train.sub                                    # the baseline
squeue -u $USER
tail -f logs/cone-p2-1280-<jobid>.out
```

Smoke test first — one epoch on 2 % of the data, which should land in a few
minutes:

```bash
sbatch --export=ALL,RUN_NAME=smoke,EPOCHS=1,FRACTION=0.02,CHAIN=0 \
  scripts/slurm/train.sub
```

Task 7 (what 640/960 cost against the 1280 baseline):

```bash
sbatch --export=ALL,RUN_NAME=p2_960,IMGSZ=960  scripts/slurm/train.sub
sbatch --export=ALL,RUN_NAME=p2_640,IMGSZ=640  scripts/slurm/train.sub
```

## What the script handles

- **The absolute-`path:` trap.** `configs/cone.yaml` holds a Mac path, and
  ultralytics resolves a relative `path:` against the CWD, not the yaml. The
  job writes its own `runs/<name>/cone.cluster.yaml` pointing at wherever the
  data actually landed. Never hand-edit the checked-in yaml — `merge.py` owns it.
- **Walltime overruns.** 100 epochs @1280 will not fit in a 4 h standby slot.
  Training is wrapped in `timeout` with a 10 min margin so the script's tail
  always runs; it then resubmits itself and `train.py --resume` picks up
  `last.pt`. `CHAIN_MAX=12` bounds the chain. `CHAIN=0` turns it off.
- **Caches off `$HOME`.** `YOLO_CONFIG_DIR`, `MPLCONFIGDIR`, `TORCH_HOME` and
  `HF_HOME` all point into scratch; home is capped at 25 GB.
- **Node-local staging.** `STAGE_LOCAL=1` copies the dataset to `$TMPDIR`
  first, which is worth the few minutes at 1280 where the dataloader binds.
  Set `STAGE_LOCAL=0` to read straight from scratch.
- **Eval.** On a clean finish it runs `eval_buckets.py` against `best.pt` and
  appends the bucketed recall to `results/runs.csv` under the same run name.

## Multi-GPU

Uncomment `#SBATCH --gres=gpu:4` in `train.sub`, then:

```bash
sbatch --export=ALL,DEVICE=0,1,2,3,BATCH=32 scripts/slurm/train.sub
```

Keep `--ntasks=1` — ultralytics spawns its own DDP workers, and wrapping it in
`srun` with several tasks starts N independent trainings instead. Set `BATCH`
explicitly; AutoBatch (`-1`) is unreliable under DDP.

## Not Gilbreth?

- **Gautschi / Negishi** — same layout; check `slist` for the partition name.
- **Anvil (ACCESS)** — partitions are `gpu` / `gpu-debug`, and it wants
  `--gpus-per-node=1` rather than `--gres=gpu:1`.
