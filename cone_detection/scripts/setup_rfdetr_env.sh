#!/usr/bin/env bash
# Create the SEPARATE rfdetr env. Never installs into `cone` - that env is
# what env_check.py validates (torch 2.14.0 / ultralytics 8.4.155) and rfdetr
# pins its own torch.
#
#   bash scripts/setup_rfdetr_env.sh
set -euo pipefail

CONDA_BASE="${CONDA_BASE:-$(conda info --base 2>/dev/null)}"
[[ -n "$CONDA_BASE" ]] || { echo "conda not found - install miniforge or set CONDA_BASE" >&2; exit 1; }
ENV_NAME="${ENV_NAME:-rfdetr}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "==> creating conda env '$ENV_NAME' (python 3.11)"
"$CONDA_BASE/bin/conda" create -y -n "$ENV_NAME" python=3.11

PY="$CONDA_BASE/envs/$ENV_NAME/bin/python"
echo "==> installing $ROOT/requirements-rfdetr.txt"
"$PY" -m pip install --upgrade pip
"$PY" -m pip install -r "$ROOT/requirements-rfdetr.txt"

echo
echo "==> verifying"
"$PY" - <<'PY'
import torch, rfdetr
archs = sorted(n for n in dir(rfdetr) if n.startswith("RFDETR"))
print(f"  torch    {torch.__version__}  cuda={torch.cuda.is_available()} "
      f"mps={getattr(torch.backends,'mps',None) and torch.backends.mps.is_available()}")
print(f"  rfdetr   {getattr(rfdetr,'__version__','?')}")
print(f"  archs    {archs}")
PY

echo
echo "==> confirming the cone env is untouched"
"$CONDA_BASE/envs/cone/bin/python" "$ROOT/scripts/env_check.py" | tail -3
