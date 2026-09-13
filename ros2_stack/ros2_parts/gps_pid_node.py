#!/usr/bin/env python3
"""ROS 2 port of parts/gps_pid.py.

Minimal GPS path follower:
1. Take the local XY pose from gps_to_xy_node (the part projected the fix
   itself, which is the same thing).
2. Pick a lookahead point on the path.
3. PID on heading error to that point.
4. Publish steering plus constant throttle.

subscribes: gps/pose (PoseStamped)
publishes:  cmd/steering (Float64, normalized -1 to 1), cmd/throttle (Float64)
"""

import math
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64

from ros2_parts.orientation import normalize_angle_rad, quaternion_to_yaw
from ros2_parts.parameters import declare

R_EARTH_M = 6378137.0


def _find_column(dtype_names, candidates):
    lower_map = {name.lower(): name for name in dtype_names}
    for candidate in candidates:
        key = candidate.lower()
        if key in lower_map:
            return lower_map[key]
    raise ValueError(f"Could not find any of columns {candidates} in {dtype_names}")


class GpsPidNode(Node):
    def __init__(self):
        super().__init__("gps_pid")

        path_csv = declare(
            self, "path_csv", "", "Waypoint CSV with latitude and longitude columns.")
        self.lookahead_m = declare(
            self, "lookahead_m", 6.0, "Distance ahead along the path to aim for.")
        self.throttle = declare(self, "throttle", 0.22, "Constant throttle returned.")
        self.kp = declare(self, "kp", 1.0, "Proportional gain on heading error.")
        self.ki = declare(self, "ki", 0.0, "Integral gain on heading error.")
        self.kd = declare(self, "kd", 0.15, "Derivative gain on heading error.")
        self.max_steer_deg = declare(
            self, "max_steer_deg", 35.0, "Steering angle at full lock, in degrees.")
        self.search_window = declare(
            self, "search_window", 20, "Waypoints searched ahead of the last match.")

        self.path_csv = Path(path_csv)
        self.max_steer_rad = math.radians(self.max_steer_deg)
        self.last_closest_idx = None

        self.lat_path, self.lon_path = self._load_latlon_path(self.path_csv)
        self.ref_lat_deg = float(self.lat_path[0])
        self.ref_lon_deg = float(self.lon_path[0])
        self.ref_lat_rad = math.radians(self.ref_lat_deg)
        self.ref_lon_rad = math.radians(self.ref_lon_deg)
        self.cos_ref_lat = max(math.cos(self.ref_lat_rad), 1e-6)

        self.path_x_m, self.path_y_m = self._latlon_to_local_xy(self.lat_path, self.lon_path)
        self.path_s_m = self._build_path_s(self.path_x_m, self.path_y_m)
        self.path_heading_rad = self._build_path_heading(self.path_x_m, self.path_y_m)
        self.path_length_m = float(self.path_s_m[-1]) if len(self.path_s_m) else 0.0

        self.error_integral = 0.0
        self.last_error = 0.0
        self.last_time = None

        self.get_logger().info(
            f"Loaded {len(self.path_x_m)} points from {self.path_csv} "
            f"with lookahead={self.lookahead_m:.2f} m")

        self.steer_pub = self.create_publisher(Float64, "cmd/steering", 10)
        self.throt_pub = self.create_publisher(Float64, "cmd/throttle", 10)
        self.create_subscription(PoseStamped, "gps/pose", self.on_pose, 10)

    def _load_latlon_path(self, csv_path):
        data = np.genfromtxt(
            csv_path,
            delimiter=",",
            names=True,
            comments="#",
            autostrip=True,
            dtype=float,
            encoding="utf-8",
        )

        if data.dtype.names is None:
            raise ValueError(f"Failed to parse path CSV header from {csv_path}")

        lat_col = _find_column(data.dtype.names, ("latitude", "lat", "lat_deg"))
        lon_col = _find_column(data.dtype.names, ("longitude", "lon", "lon_deg"))

        lat = np.asarray(data[lat_col], dtype=float)
        lon = np.asarray(data[lon_col], dtype=float)
        if lat.size < 2:
            raise ValueError(f"Path must contain at least 2 points: {csv_path}")
        return lat, lon

    def _latlon_to_local_xy(self, lat_deg, lon_deg):
        lat_rad = np.radians(lat_deg)
        lon_rad = np.radians(lon_deg)
        dlat = lat_rad - self.ref_lat_rad
        dlon = lon_rad - self.ref_lon_rad
        x_east = dlon * self.cos_ref_lat * R_EARTH_M
        y_north = dlat * R_EARTH_M
        return np.asarray(x_east, dtype=float), np.asarray(y_north, dtype=float)

    def _build_path_s(self, path_x, path_y):
        segment_lengths = np.hypot(np.diff(path_x), np.diff(path_y))
        return np.concatenate(([0.0], np.cumsum(segment_lengths)))

    def _build_path_heading(self, path_x, path_y):
        dx = np.diff(path_x)
        dy = np.diff(path_y)
        heading = np.arctan2(dy, dx)
        return np.concatenate((heading, [heading[-1]]))

    def _find_closest_idx(self, x_m, y_m):
        distances = np.hypot(self.path_x_m - x_m, self.path_y_m - y_m)
        return int(np.argmin(distances))

    def _find_closest_idx_forward(self, x_m, y_m):
        # searching a window ahead rather than the whole path keeps the kart
        # from snapping backwards where the path crosses itself
        n = len(self.path_x_m)
        if self.last_closest_idx is None:
            best = self._find_closest_idx(x_m, y_m)
        else:
            window = np.arange(
                self.last_closest_idx, self.last_closest_idx + self.search_window + 1) % n
            distances = np.hypot(self.path_x_m[window] - x_m, self.path_y_m[window] - y_m)
            best = int(window[int(np.argmin(distances))])
        self.last_closest_idx = best
        return best

    def reset_path_tracking(self):
        self.last_closest_idx = None

    def _find_target_idx(self, closest_idx):
        target_s_m = self.path_s_m[closest_idx] + self.lookahead_m
        if target_s_m <= self.path_s_m[-1]:
            return int(np.searchsorted(self.path_s_m, target_s_m, side="left"))

        wrapped_s_m = target_s_m - self.path_s_m[-1]
        return int(np.searchsorted(self.path_s_m, wrapped_s_m, side="left"))

    def _compute_dt(self):
        current_time = time.time()
        if self.last_time is None:
            dt = 0.05
        else:
            dt = max(current_time - self.last_time, 1e-3)
        self.last_time = current_time
        return dt

    def compute_control_from_xy(self, x_m, y_m, yaw_rad):
        closest_idx = self._find_closest_idx_forward(float(x_m), float(y_m))
        target_idx = self._find_target_idx(closest_idx)

        target_x_m = float(self.path_x_m[target_idx])
        target_y_m = float(self.path_y_m[target_idx])
        dx_m = target_x_m - float(x_m)
        dy_m = target_y_m - float(y_m)

        desired_heading_rad = math.atan2(dy_m, dx_m)
        heading_error_rad = normalize_angle_rad(desired_heading_rad - float(yaw_rad))

        dt = self._compute_dt()
        self.error_integral += heading_error_rad * dt
        derivative = (heading_error_rad - self.last_error) / dt
        self.last_error = heading_error_rad

        steering_angle_rad = (
            self.kp * heading_error_rad
            + self.ki * self.error_integral
            + self.kd * derivative
        )
        steering_cmd = float(np.clip(steering_angle_rad / self.max_steer_rad, -1.0, 1.0))

        self.get_logger().debug(
            f"closest_idx={closest_idx}, target_idx={target_idx}, "
            f"heading_error={math.degrees(heading_error_rad):.1f} deg, "
            f"steering_cmd={steering_cmd:.3f}")
        return steering_cmd, self.throttle

    def run(self, x_m, y_m, yaw_rad):
        return self.compute_control_from_xy(float(x_m), float(y_m), float(yaw_rad))

    def on_pose(self, pose):
        steering_cmd, throttle_cmd = self.run(
            pose.pose.position.x,
            pose.pose.position.y,
            quaternion_to_yaw(pose.pose.orientation))

        self.steer_pub.publish(Float64(data=float(steering_cmd)))
        self.throt_pub.publish(Float64(data=float(throttle_cmd)))


def main(args=None):
    rclpy.init(args=args)
    node = GpsPidNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
