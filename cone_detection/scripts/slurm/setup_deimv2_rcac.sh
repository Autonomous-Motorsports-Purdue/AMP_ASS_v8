#!/usr/bin/env bash
# One-time deimv2 env on an RCAC GPU cluster. Run this ON THE CLUSTER, on a
# GPU node (or an interactive session) so the CUDA build matches the driver.
#
#   sinteractive -A <account> -p <queue> --gres=gpu:1 -t 00:30:00
#   bash scripts/slurm/setup_deimv2_rcac.sh
set -euo pipefail

ENV_NAME="${ENV_NAME:-deimv2}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Keep the clone on scratch, not in $HOME (25 GB quota) - the backbone
# checkpoints land next to it under ckpts/.
DEIMV2_HOME="${DEIMV2_HOME:-${RCAC_SCRATCH:-$ROOT}/DEIMv2}"
REPO="${REPO:-https://github.com/Intellindust-AI-Lab/DEIMv2.git}"

module --force purge
module load modtree/gpu 2>/dev/null || true
module load conda
module load cuda || true
eval "$(conda shell.bash hook)"

echo "==> driver / CUDA on this node"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true

echo "==> clone: $DEIMV2_HOME"
if [[ -d "$DEIMV2_HOME/.git" ]]; then
  echo "    already there"
else
  git clone --depth 1 "$REPO" "$DEIMV2_HOME"
fi

conda create -y -n "$ENV_NAME" python=3.11
conda activate "$ENV_NAME"
python -m pip install --upgrade pip

# CUDA torch FIRST, at DEIMv2's pin, so nothing pulls a CPU wheel over it.
# cu124 suits recent Ampere/Hopper drivers; check nvidia-smi above and switch
# to cu121 if the driver is older.
CUDA_CHANNEL="${CUDA_CHANNEL:-cu124}"
python -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url "https://download.pytorch.org/whl/${CUDA_CHANNEL}"

# Then the rest of upstream's requirements, WITHOUT letting it re-resolve torch.
grep -viE '^(torch|torchvision)==' "$DEIMV2_HOME/requirements.txt" \
  > /tmp/deimv2-req-$$.txt
python -m pip install -r /tmp/deimv2-req-$$.txt
rm -f /tmp/deimv2-req-$$.txt
python -m pip install pillow numpy pyyaml

echo
echo "==> verifying"
DEIMV2_HOME="$DEIMV2_HOME" PROJECT_ROOT="$ROOT" python - <<'PY'
import os, sys
import torch
home, root = os.environ["DEIMV2_HOME"], os.environ["PROJECT_ROOT"]
sys.path.insert(0, home)
sys.path.insert(0, os.path.join(root, "scripts"))
print(f"  torch  {torch.__version__}  cuda={torch.cuda.is_available()}  n={torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"  gpu    {torch.cuda.get_device_name(0)}")
from engine.core import YAMLConfig            # noqa: F401
print("  engine engine.core imports OK")
import deimv2_cfg
print(f"  clone  {deimv2_cfg.deimv2_home()}")
print(f"  step   {deimv2_cfg.RESOLUTION_STEP} -> 1280 valid for every variant")
PY

echo
echo "put this in your ~/.bashrc (train_deimv2.sub also sets it):"
echo "  export DEIMV2_HOME=$DEIMV2_HOME"
echo
echo "next:"
echo "  python scripts/yolo_to_coco.py             # COCO mirror on scratch"
echo "  sbatch scripts/slurm/train_deimv2.sub"
