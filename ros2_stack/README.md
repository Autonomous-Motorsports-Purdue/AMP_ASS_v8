# ros2_parts

ROS 2 (Humble / Ubuntu 22.04) port of the Donkeycar stack in [`parts/`](../parts).
Every part is now a plain `rclpy` node speaking standard ROS interfaces.
Nothing in this package imports Donkeycar.

## Shape of a node

Each file is one `rclpy.node.Node` subclass that opens its own hardware,
subscribes to what it needs and publishes real messages. There is no framework
layer between the node and ROS — what you see in the file is what runs:

```python
class ImuNode(Node):

    def __init__(self) -> None:
        super().__init__('imu')
        self.port_name = declare(self, 'port', '/dev/ttyACM1', 'Serial port the IMU is on.')
        self.serial, self.serial_io = self.open_serial()
        self.publisher = self.create_publisher(Imu, 'imu/data', qos_profile_sensor_data)
        self.create_timer(1.0 / rate_hz, self.publish_sample)
```

Three small modules are shared, because copying them 42 times would be worse:
[`parameters.py`](ros2_parts/parameters.py) (one helper for described
parameters), [`orientation.py`](ros2_parts/orientation.py) (quaternion and
angle conversions) and [`geometry.py`](ros2_parts/geometry.py) (the cross-track
error). All three are plain functions over the standard API.

Sensor nodes publish on a timer; everything downstream is event-driven off its
primary input, except the filters and loggers, which sample the latest of
several inputs on a timer.

## Interfaces

Sensors, poses and images use the standard messages, so the graph works with
`rviz2`, `ros2 bag`, `tf2` and `robot_localization` as-is:

| Topic | Type |
| --- | --- |
| `imu/data` | `sensor_msgs/Imu` |
| `gps/fix`, `gps/fused_fix`, `gps/ekf_fix` | `sensor_msgs/NavSatFix` |
| `gps/vel`, `gps/ekf_vel` | `geometry_msgs/TwistStamped` |
| `gps/pose`, `predicted_pose` | `geometry_msgs/PoseStamped` |
| `odometry/filtered` | `nav_msgs/Odometry` |
| `camera/*/image_raw`, `perception/*_mask`, `perception/*_image` | `sensor_msgs/Image` |
| `camera/left/camera_info` | `sensor_msgs/CameraInfo` |
| `perception/centroid`, `perception/waypoint`, `perception/waypoint_ground` | `geometry_msgs/PointStamped` |
| `perception/midline` | `nav_msgs/Path` |

Actuator commands stay scalar on purpose. The kart takes a normalised steering
value and a throttle in eRPM, and there is no honest conversion to
`geometry_msgs/Twist` (which is a velocity) or `ackermann_msgs/AckermannDrive`
(which wants a steering angle in radians and a speed in m/s). Dressing them up
as either would misstate the units:

| Topic | Type | Meaning |
| --- | --- | --- |
| `cmd/steering`, `cmd/auto_steering`, `cmd/user_steering` | `std_msgs/Float64` | normalised, -1 (left) to 1 (right) |
| `cmd/throttle`, `cmd/auto_throttle`, `cmd/user_throttle` | `std_msgs/Float64` | eRPM, or normalised from the camera stack |
| `cmd/commanded_steering`, `cmd/commanded_throttle` | `std_msgs/Float64` | what the UART actually sent |
| `safety/heartbeat` | `std_msgs/Bool` | false stops the kart |
| `gps/fix_type` | `std_msgs/String` | `RTK FIXED` etc., which `NavSatStatus` cannot express |

## Build

```bash
mkdir -p ~/amp_ws/src
ln -s /path/to/AMP_ASS_v8/ros2_stack ~/amp_ws/src/ros2_parts

cd ~/amp_ws
rosdep install --from-paths src --ignore-src -r -y
pip install -r src/ros2_parts/requirements.txt
colcon build --packages-select ros2_parts
source install/setup.bash
```

Each node only imports what it needs, so a machine that never runs the ZED,
ONNX or GStreamer nodes does not need those libraries.

## Run

```bash
# The whole CTE stack, the way cte_runner.py wires it
ros2 launch ros2_parts cte.launch.py path_csv:=/abs/path/to/path_xy_throttle.csv

# Or one node at a time
ros2 run ros2_parts gps --ros-args -p port:=/dev/ttyACM0
ros2 topic echo /gps/fix
```

Launch files mirror the runners: [`cte.launch.py`](launch/cte.launch.py)
(`cte_runner.py`), [`gps_pid.launch.py`](launch/gps_pid.launch.py)
(`gps_pid_runner.py`) and [`pursuit.launch.py`](launch/pursuit.launch.py)
(`pursuit_runner.py`), each with a matching file in [`config/`](config/).

## Nodes

`ros2 run ros2_parts <executable>`. Every node takes `rate_hz` where it drives
a loop, plus whatever `ros2 param describe` lists.

| Part | Executable | Subscribes | Publishes |
| --- | --- | --- | --- |
| `bno086.py` | `bno086` | — | `imu/data` |
| `control_mux.py` | `control_mux` | user + auto commands | `cmd/steering`, `cmd/throttle` |
| `controller.py` | `mpc_part` | `gps/pose` | `cmd/desired_yaw_rate`, `cmd/desired_speed` |
| `controller.py` | `closed_loop_controller` | desired rate, `imu/data` | `cmd/steering`, `cmd/throttle` |
| `cte_controller.py` | `cte_controller` | `odometry/filtered`, `gps/pose`, `gps/vel` | commands, `controller/cross_track_error`, `controller/waypoint_index` |
| `curve_fit.py` | `curve_fit` | `perception/drivable_mask` | `perception/waypoint`, `perception/curve_image` |
| `ekf_localizer.py` | `ekf_localizer` | `gps/fix`, `imu/data` | `gps/ekf_fix`, `gps/ekf_vel` |
| `fake_gps.py` | `fake_gps` | — | `gps/fix`, `gps/fix_type` |
| `fake_imu.py` | `fake_imu` | — | `imu/data` |
| `frame_publisher.py` | `frame_publisher` | — | `camera/{left,right}/image_raw` |
| `gps.py` | `gps` | — | `gps/fix`, `gps/vel`, `gps/fix_type`, quality scalars |
| `gps_csv.py` | `gps_csv` | — | the same, replayed from a CSV |
| `gps_pid.py` | `gps_pid` | `gps/pose` | `cmd/steering`, `cmd/throttle` |
| `gps_to_xy.py` | `gps_to_xy` | `gps/fix` | `gps/pose`, optional `/tf` |
| `gps_to_xy.py` | `xy_to_gps` | `odometry/filtered` | `gps/fused_fix` |
| `gps_visualizer.py` | `gps_visualizer` | `gps/fix`, `imu/data` | — |
| `gstreamer_video_sync.py` | `gstreamer_video_sync` | `loop/*` | `video/*` |
| `heading_fusion.py` | `heading_fusion` | `gps/pose`, `gps/vel`, `imu/data` | `odometry/filtered` |
| `health_check.py` | `health_check` | — | `safety/heartbeat` |
| `image_cv.py` | `image_cv` | `camera/image_raw` | detection image, centroid, area |
| `image_publisher.py` | `image_publisher` | — | `camera/image_raw` |
| `imu.py` | `imu` | — | `imu/data` |
| `imu_visualizer.py` | `imu_visualizer` | `imu/data` | — |
| `lane_detect.py` | `lane_detect` | `camera/left/image_raw` | `perception/midline`, annotated image |
| `log.py` | `log` | object centroid, area | — |
| `logger.py` | `logger` | frames + commands | — |
| `logger2.py` | `logger2` | any `Float64` topics (`topics` param) | — |
| `logger_gps.py` | `logger_gps` | the whole run | — |
| `loop_clock.py` | `loop_clock` | — | `loop/index`, `loop/monotonic_ns` |
| `onnx.py` | `onnx` | `camera/image_raw` | `perception/{lane,drivable}_mask` |
| `predictive_model.py` | `predictive_model` | `gps/pose`, `gps/vel`, `imu/data` | `predicted_pose` |
| `preprocessor.py` | `preprocessor` | `camera/image_raw` | `camera/image_preprocessed` |
| `pure_pursuit.py` | `pure_pursuit` | `perception/waypoint_ground` | `cmd/auto_{steering,throttle}` |
| `pure_pursuit_controller.py` | `pure_pursuit_controller` | `odometry/filtered` | commands, `controller/pure_pursuit_debug` |
| `segment_model.py` | `segment_model` | `camera/left/image_raw` | segmented image, centroid, commands |
| `test.py` | `test` | — | `cmd/user_{steering,throttle}` |
| `threaded_socket_pub_part.py` | `threaded_socket_pub` | fixes, odometry, IMU, steering | — (TCP) |
| `translate.py` | `translate` | `perception/waypoint` | `perception/waypoint_ground` |
| `uart.py` | `uart` | commands, heartbeat | — |
| `uart_backup.py` | `uart_backup` | commands, heartbeat, `gps/fix_type` | `cmd/commanded_*` |
| `uart_backup2.py` | `uart_backup2` | `cmd/raw_throttle` | — (bench rig) |
| `zed_frame_publisher.py` | `zed_frame_publisher` | — | ZED images + `camera_info` |

## Behaviour changes worth knowing

The algorithms are unchanged. These are the places where moving onto standard
messages changed a number, and each one can affect how the kart drives:

* **Yaw rate is now radians per second.** `sensor_msgs/Imu` specifies that
  unit; the Donkeycar IMU part passed the board's degrees per second straight
  through, and `ClosedLoopController` subtracted it from the MPC's radians per
  second. That mismatch is gone, which means **the closed-loop gains need
  retuning** — the error term is now about 57x smaller.
* **The EKF gets its heading in degrees.** `gps_mpc_runner.py` fed
  `ekf_localizer` the IMU part's yaw in radians into an argument the filter
  converted as degrees, so its body-to-world rotation was wrong. It is now
  taken from the orientation quaternion in the right units.
* **IMU accuracy rides in `orientation_covariance`.** `heading_fusion` reads
  its rejection threshold from there rather than from a separate topic.
* **The health check can actually check.** The part had a real socket check
  behind an unconditional `return True`. It is implemented and gated behind
  `enabled`, which defaults to false so the shipped behaviour is unchanged.
* **The UART nodes fail safe.** A heartbeat or fix topic that has never been
  published now stops the kart, rather than being read as a missing value.

Smaller notes:

* `curve_fit` and `lane_detect` each took an input their bodies never read (the
  lane mask and the depth frame); those subscriptions are gone.
* `cte_controller` computed a path-curvature feed-forward that was commented
  out of the final steering, and `segment_model` computed an offset-to-throttle
  curve it then overrode with a constant. Neither took effect, so neither is
  carried over.
* `logger2` now names its topics by parameter. For anything you intend to keep,
  `ros2 bag record` is the better tool.
* `bno086`'s `wrap360` added 361 degrees to negative angles. Python's modulo
  never produces one, so the branch was dead; it is 360 here.

## The cross-track sign

`cte_controller` was the last thing importing Donkeycar, for `la.Line3D`,
`la.Vec3` and `utils.dist`. Those are now
[`geometry.py`](ros2_parts/geometry.py).

The sign matters — it decides which way the kart steers off-path — and the
Donkeycar submodule is not checked out here, so it was derived rather than
copied: positive error means the kart is left of the path, which the PID turns
into a rightward correction. `cross_track_error` reproduces the original's 3D
embedding exactly on random inputs, and the convention is pinned by
[`test_geometry.py`](test/test_geometry.py). **Confirm it on the kart at low
speed before trusting it at pace.**

## Test

```bash
colcon test --packages-select ros2_parts && colcon test-result --verbose
```

The tests cover the two shared modules the nodes depend on for correctness: the
path geometry and the orientation conversions.
