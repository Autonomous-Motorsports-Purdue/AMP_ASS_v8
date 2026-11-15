#!/usr/bin/env bash
# Run INSIDE the container. Builds the TensorRT FP16 engine on this Jetson -
# an engine only runs on the device and TensorRT version that built it, so it
# cannot be made on the Mac. Takes several minutes; writes the .engine next
# to the .pt (/cone/models/base_p2_1280.engine). Engines are gitignored.
#
#   bash /cone/scripts/jetson/export_engine.sh                  # 736x1280: a 16:9 camera (Brio 105)
#   H=1280 bash /cone/scripts/jetson/export_engine.sh           # square 1280, as the eval ran
#
# 736x1280 does the same scaling as square 1280 for a 16:9 frame (1920x1080
# -> 1280x720) but skips the ~40% of the square that would be padding.
set -euo pipefail
PT="${1:-/cone/models/base_p2_1280.pt}"
H="${H:-736}"
W="${W:-1280}"

python3 - "$PT" "$H" "$W" <<'EOF'
import sys
from ultralytics import YOLO
pt, h, w = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
out = YOLO(pt).export(format="engine", imgsz=(h, w), half=True, batch=1, device=0)
print(f"\nengine: {out}  (input {h}x{w}; pass --imgsz {h} {w} to camera_loop.py)")
EOF
