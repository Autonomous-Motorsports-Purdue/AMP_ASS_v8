#!/usr/bin/env bash
# Run ON THE JETSON (the host, not inside Docker). Starts the Ultralytics
# JetPack 6 container with this repo mounted at /cone, the webcam passed
# through, and the screen too if a desktop session is running (for --show).
#
#   bash scripts/jetson/run_container.sh             # from the repo root
#   CAM=/dev/video2 bash scripts/jetson/run_container.sh
set -euo pipefail

DIR="${CONE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CAM="${CAM:-/dev/video0}"
IMAGE="${IMAGE:-ultralytics/ultralytics:latest-jetson-jetpack6}"

[[ -e "$CAM" ]] || { echo "no camera at $CAM - see: ls /dev/video*" >&2; exit 1; }

# The Orin Nano Super modes need JetPack 6.2+ (L4T R36.4+).
rel="$(head -1 /etc/nv_tegra_release 2>/dev/null || echo unknown)"
echo "L4T: $rel"
grep -q "R36" <<<"$rel" || echo "[warn] not JetPack 6 - this image will not match. Use the jetpack5 tag or update the board."

# A throttled board gives a misleadingly slow number.
echo "power mode: $(sudo nvpmodel -q 2>/dev/null | head -1)  <- want MAXN SUPER before timing"

X=()
if [[ -n "${DISPLAY:-}" ]]; then
  xhost +local:root >/dev/null 2>&1 || true
  X=(-e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix)
fi

exec sudo docker run -it --rm --ipc=host --runtime=nvidia \
  -v "$DIR":/cone -e CONE_HOST_DIR="$DIR" --device "$CAM" "${X[@]}" "$IMAGE" bash
