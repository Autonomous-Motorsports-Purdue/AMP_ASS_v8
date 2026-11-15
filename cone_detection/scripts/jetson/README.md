# Running the cone detector on the Jetson

Target: Jetson Orin Nano Super (JetPack 6), Logitech Brio 105 webcam.
The model to deploy is `models/base_p2_1280.pt` (tracked in git, 6 MB).

## Once per Jetson

```bash
cat /etc/nv_tegra_release          # want R36.4+ (JetPack 6.2) for the Super power modes
sudo nvpmodel -q --verbose         # pick MAXN SUPER: sudo nvpmodel -m <N>
sudo jetson_clocks
sudo apt install -y v4l-utils
v4l2-ctl --list-devices            # find the Brio's /dev/videoN
git clone https://github.com/Autonomous-Motorsports-Purdue/AMP_ASS_v8.git && cd AMP_ASS_v8/cone_detection
```

## Every time

```bash
cd AMP_ASS_v8/cone_detection && git pull
bash scripts/jetson/run_container.sh                       # host -> container shell
# inside the container:
bash /cone/scripts/jetson/export_engine.sh                 # first time / new model only
python3 /cone/scripts/jetson/camera_loop.py --seconds 30   # time it
python3 /cone/scripts/jetson/camera_loop.py --save /cone/run.mp4 --seconds 20   # boxes -> cone_detection/run.mp4
```

`camera_loop.py` should report `camera: 1920x1080 @ 30 fps, MJPG`. If it
reports anything smaller, distant cones will be missed. Check the modes with
`v4l2-ctl -d /dev/video0 --list-formats-ext`.

The loop is capped at 30 Hz by the camera. The summary's "model alone could
do N Hz" line is the detector's own speed.
