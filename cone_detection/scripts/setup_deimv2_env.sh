#!/usr/bin/env bash
# Clone DEIMv2 and create the SEPARATE deimv2 env.
#
#   bash scripts/setup_deimv2_env.sh
#
# Never installs into `cone` (env_check.py validates that one: torch 2.14.0 /
# ultralytics 8.4.155) or `rfdetr`. DEIMv2 pins torch==2.5.1.
#
# DEIMv2 is a repo, not a package: the clone IS the install. $DEIMV2_HOME points
# at it, and scripts/deimv2_cfg.py reads that variable, so a cluster can keep
# the clone on scratch instead of here.
set -euo pipefail

CONDA_BASE="${CONDA_BASE:-$(conda info --base 2>/dev/null)}"
[[ -n "$CONDA_BASE" ]] || { echo "conda not found - install miniforge or set CONDA_BASE" >&2; exit 1; }
ENV_NAME="${ENV_NAME:-deimv2}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEIMV2_HOME="${DEIMV2_HOME:-$ROOT/deimv2/DEIMv2}"
REPO="${REPO:-https://github.com/Intellindust-AI-Lab/DEIMv2.git}"

echo "==> clone: $DEIMV2_HOME"
if [[ -d "$DEIMV2_HOME/.git" ]]; then
  echo "    already there; leaving it alone (git -C \"$DEIMV2_HOME\" pull to update)"
else
  mkdir -p "$(dirname "$DEIMV2_HOME")"
  git clone --depth 1 "$REPO" "$DEIMV2_HOME"
fi

echo "==> creating conda env '$ENV_NAME' (python 3.11)"
"$CONDA_BASE/bin/conda" create -y -n "$ENV_NAME" python=3.11

PY="$CONDA_BASE/envs/$ENV_NAME/bin/python"
"$PY" -m pip install --upgrade pip

# Upstream's own pins first - that file is the source of truth for the model.
echo "==> installing $DEIMV2_HOME/requirements.txt (upstream pins)"
"$PY" -m pip install -r "$DEIMV2_HOME/requirements.txt"

echo "==> installing $ROOT/requirements-deimv2.txt (extras for the shared scripts)"
"$PY" -m pip install -r "$ROOT/requirements-deimv2.txt"

echo
echo "==> verifying"
DEIMV2_HOME="$DEIMV2_HOME" PROJECT_ROOT="$ROOT" "$PY" - <<'PY'
import os, sys
import torch

home, root = os.environ["DEIMV2_HOME"], os.environ["PROJECT_ROOT"]
sys.path.insert(0, home)                      # the clone, for engine.*
sys.path.insert(0, os.path.join(root, "scripts"))

mps = getattr(torch.backends, "mps", None)
print(f"  torch    {torch.__version__}  cuda={torch.cuda.is_available()} "
      f"mps={bool(mps and mps.is_available())}")
from engine.core import YAMLConfig             # noqa: F401
print("  engine   engine.core imports OK")

import deimv2_cfg
print(f"  clone    {deimv2_cfg.deimv2_home()}")
print(f"  variants {sorted(deimv2_cfg.VARIANTS)}")
print(f"  step     {deimv2_cfg.RESOLUTION_STEP} -> 1280 is valid for all of them")
missing = [a for a, c in deimv2_cfg.NEEDS_CKPT.items()
           if not (deimv2_cfg.deimv2_home() / c).exists()]
if missing:
    print(f"  ckpts    {missing} still need a manual backbone download "
          f"(see deimv2/README.md); the hgnetv2_* variants do not")
PY

echo
echo "==> confirming the cone env is untouched"
"$CONDA_BASE/envs/cone/bin/python" "$ROOT/scripts/env_check.py" | tail -3

echo
echo "next:"
echo "  python scripts/yolo_to_coco.py          # shared with rf-detr; skip if built"
echo "  python scripts/train_deimv2.py --dry-run"
