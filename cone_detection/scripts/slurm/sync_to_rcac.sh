#!/usr/bin/env bash
# Push the code and the merged dataset to RCAC scratch.
#
#   ./scripts/slurm/sync_to_rcac.sh <user>@gilbreth.rcac.purdue.edu
#   CODE_ONLY=1 ./scripts/slurm/sync_to_rcac.sh <user>@gilbreth.rcac.purdue.edu
#
# data/merged is hardlinked into data/interim + data/raw, so it reads as 56 MB
# locally but dereferences to ~5.7 GB. That is what actually crosses the wire,
# and it is why the destination is scratch and not $HOME (25 GB quota).
set -euo pipefail

HOST="${1:?usage: sync_to_rcac.sh user@cluster.rcac.purdue.edu}"
REMOTE="${REMOTE:-\$RCAC_SCRATCH/aks-cone-detection}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# $RCAC_SCRATCH has to expand on the far side, not here - and only a login
# shell defines it; a bare `ssh host cmd` leaves it empty, which would turn
# REMOTE into /aks-cone-detection.
DEST="$(ssh "$HOST" "bash -lc 'test -n \"\$RCAC_SCRATCH\" && mkdir -p $REMOTE && cd $REMOTE && pwd'")"
[[ -n "$DEST" ]] || { echo "could not resolve \$RCAC_SCRATCH on $HOST" >&2; exit 1; }
echo "destination: $HOST:$DEST"

echo "--- code ---"
rsync -avh --progress \
  --exclude '.git' --exclude 'data' --exclude 'runs' --exclude '__pycache__' \
  --exclude '*.pyc' --exclude 'logs' --exclude 'deimv2/DEIMv2' \
  "$ROOT/" "$HOST:$DEST/"

if [[ "${CODE_ONLY:-0}" == "1" ]]; then
  echo "code only - skipping dataset"; exit 0
fi

echo "--- dataset: YOLO (~5.7 GB, resumable: just rerun) ---"
rsync -avh --progress --partial \
  "$ROOT/data/merged/" "$HOST:$DEST/data/merged/"

# The COCO mirror is hardlinked to data/merged locally, so syncing it too would
# send the same 5.7 GB twice. Rebuild it on the cluster instead - it is cheap
# and guarantees the two mirrors stay byte-identical.
echo "--- COCO mirror: rebuilding remotely (not transferred twice) ---"
ssh "$HOST" "cd $DEST && ls scripts/yolo_to_coco.py >/dev/null" \
  && echo "  run on the cluster:  python scripts/yolo_to_coco.py"

echo
echo "done. On the cluster:"
echo "  cd $DEST && sbatch scripts/slurm/train.sub"
