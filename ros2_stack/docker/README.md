# Dev container

Ubuntu 22.04 / ROS 2 Humble, in two flavours from one
[`Dockerfile`](Dockerfile):

| | `laptop` (default) | `jetson` |
| --- | --- | --- |
| Base | `ros:humble` (amd64 + arm64) | `l4t-jetpack:r36.4.0` |
| Builds on | any machine, minutes | the Jetson |
| onnxruntime | CPU | CUDA, from NVIDIA's index |
| ZED SDK / `pyzed` | no | yes |
| Hardware | none | serial, cameras, `/dev` |
| Runs | fake sensors, CSV replay, controllers, rviz2, tests | everything |

ROS 2 Humble itself **is** installed in both images, at `/opt/ros/humble` —
about 270 packages including `rclpy`, `cv_bridge`, `tf2_ros`, `rviz2` and
`colcon`. It comes from apt, which is why it never appears in
[`requirements.txt`](../requirements.txt); that file is only for pip packages.

What is not baked in is **your** package, `ros2_parts`. `ros2_stack/` is
bind-mounted over `/amp_ws/src/ros2_parts`, so you edit on the host and
`colcon build` inside.

The target is an explicit flag rather than the detected CPU arch, because an
M-series Mac is also arm64 and is not a Jetson. BuildKit only builds the stages
the chosen target needs, so a laptop build never pulls the L4T image.

## Run

From `ros2_stack/`, on any machine:

```bash
docker compose -f docker/docker-compose.yml run --rm laptop
```

On the kart:

```bash
docker compose -f docker/docker-compose.yml run --rm amp
```

Then, inside either:

```bash
colcon build --packages-select ros2_parts --symlink-install
. install/setup.bash
ros2 launch ros2_parts cte.launch.py path_csv:=/amp/gps_paths/<file>.csv
```

`--symlink-install` links the source instead of copying it, so editing a node
takes effect without another `colcon build`. You only need a plain build again
after adding an entry point to `setup.py`.

`build/`, `install/` and `log/` live in named volumes — **separate volumes per
target**, since amd64 and arm64 colcon artifacts must not mix. The repo root is
mounted at `/amp`, which is where `gps_paths/`, logs and ONNX models are.

`$AMP_TARGET` inside the container tells you which flavour you are in.

## Does it work?

```bash
docker compose -f docker/docker-compose.yml run --rm laptop bash src/ros2_parts/docker/smoke_test.sh
docker compose -f docker/docker-compose.yml run --rm amp    bash src/ros2_parts/docker/smoke_test.sh
```

[`smoke_test.sh`](smoke_test.sh) checks the ROS environment, every Python
dependency, that numpy is still 1.x, GStreamer, onnxruntime's providers, the
colcon build, a live two-node ROS graph (`fake_gps` -> `gps_to_xy` ->
`/gps/pose`) and the unit tests. On the jetson target it additionally requires
the CUDA provider, `pyzed` and visible serial devices. It exits non-zero if
anything fails, so it works as a CI gate. `--no-build` skips the colcon steps.

Expected on the laptop target: `24 passed, 0 failed, 1 skipped`. The unit tests
report one pre-existing failure (`normalize_quaternion`), which is a bug in the
package, not the container.

## The source mount is read-write

`ros2_stack/` is bind-mounted read-write, which is what makes this a dev
container: `colcon build --symlink-install` needs to link back into it. The
consequence is that anything deleting files under `/amp_ws/src/ros2_parts`
inside the container deletes them **on the host**, and Docker bind mounts do
not go through the Recycle Bin.

Commit before running containers against a tree you care about. To work from a
throwaway copy instead, point the mount somewhere else:

```yaml
    volumes:
      - ..:/amp_ws/src/ros2_parts:ro   # read-only; colcon then needs --no-symlink
```

## What does not work on the laptop target

`zed_frame_publisher` (no SDK), and anything that opens a real serial port or
camera: `gps`, `imu`, `bno086`, `uart*`, `frame_publisher`. Use `fake_gps`,
`fake_imu`, `gps_csv` and `image_publisher` instead. `onnx` and `segment_model`
run on CPU, so they are correct but slow.

## Three versions in the jetson path

Hardcoded in [`Dockerfile`](Dockerfile) — edit the line. A wrong one fails the
build, it does not fail quietly.

| What | Value | Check against |
| --- | --- | --- |
| Base image tag | `r36.4.0` | `cat /etc/nv_tegra_release` |
| ZED SDK / L4T | `5.0` / `l4t36.4` | <https://www.stereolabs.com/developers/release> |
| pip index | `jp6/cu126` | <https://pypi.jetson-ai-lab.io/jp6/> |

The container user is UID 1000, the Jetson's default. If `id -u` on the host
says otherwise, change the `useradd` line or `colcon build` cannot write into
the mounted workspace.

## Why onnxruntime is installed per target

`requirements.txt` asks for no onnxruntime at all, because the right package
differs per target. The PyPI aarch64 `onnxruntime-gpu` wheels are built against
the desktop CUDA stack and fall back to CPU on Tegra, so the laptop gets plain
CPU `onnxruntime` and the Jetson gets NVIDIA's build from
`pypi.jetson-ai-lab.io`, with an assert that `CUDAExecutionProvider` is present
so a wrong wheel is a failed build rather than a slow kart.

## GUI

`rviz2`, `gps_visualizer` and `imu_visualizer` need an X server.

- **Linux / the Jetson**: `xhost +local:docker`, and the X11 socket mount is
  already in the `amp` service.
- **Windows**: run VcXsrv with access control disabled, then
  `DISPLAY=host.docker.internal:0.0` (the compose default).
- **macOS**: XQuartz, enable "Allow connections from network clients", then
  `xhost +127.0.0.1` and `DISPLAY=host.docker.internal:0`.

On a Linux host, uncomment the `/tmp/.X11-unix` mount in the `laptop` service;
that path does not exist on Windows or macOS.

## Hardware access (jetson only)

`privileged`, `network_mode: host` and all of `/dev` are deliberate: the GPS,
IMU and VESC are USB serial adapters that get replugged and renamed, host
networking is what lets ROS 2 discovery reach the rest of the kart's network,
and `runtime: nvidia` is what provides CUDA and the ZED.
