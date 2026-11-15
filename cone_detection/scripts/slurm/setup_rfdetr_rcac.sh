#!/usr/bin/env bash
# One-time rfdetr env on an RCAC GPU cluster. Run this ON THE CLUSTER, on a
# GPU node (or an interactive session) so the CUDA build matches the driver.
#
#   sinteractive -A <account> -p <queue> --gres=gpu:1 -t 00:30:00
#   bash scripts/slurm/setup_rfdetr_rcac.sh
set -euo pipefail

ENV_NAME="${ENV_NAME:-rfdetr}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

module --force purge
module load modtree/gpu 2>/dev/null || true
module load conda
module load cuda || true
eval "$(conda shell.bash hook)"

echo "==> driver / CUDA on this node"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true

conda create -y -n "$ENV_NAME" python=3.11
conda activate "$ENV_NAME"
python -m pip install --upgrade pip

# CUDA torch FIRST, so the rfdetr install does not pull a CPU wheel over it.
# cu124 suits recent Orin/Ampere-era drivers; check nvidia-smi above and switch
# to cu121/cu126 if the driver is older/newer.
CUDA_CHANNEL="${CUDA_CHANNEL:-cu124}"
python -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/${CUDA_CHANNEL}"

# [train,loggers] is REQUIRED - the bare rfdetr package is inference only.
python -m pip install "rfdetr[train,loggers]"
python -m pip install pillow pyyaml numpy

echo
echo "==> verifying"
python - <<'PY'
import torch, rfdetr
print(f"  torch  {torch.__version__}  cuda={torch.cuda.is_available()}  n={torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"  gpu    {torch.cuda.get_device_name(0)}")
print(f"  archs  {[n for n in dir(rfdetr) if n.startswith('RFDETR') and 'Seg' not in n]}")
import sys; sys.path.insert(0, "scripts")
from backends import resolution_step
for a in ("RFDETRNano","RFDETRSmall","RFDETRBase"):
    print(f"  {a:<13} resolution step = {resolution_step(a)}")
PY

echo
echo "next:"
echo "  python scripts/yolo_to_coco.py            # build the COCO mirror on scratch"
echo "  sbatch scripts/slurm/train_rfdetr.sub"
