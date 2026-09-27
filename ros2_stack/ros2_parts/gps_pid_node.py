#!/usr/bin/env python3
# in: gps/pose (PoseStamped), path_csv (lat, lon columns) | out: cmd/steering (Float64, -1 to 1), cmd/throttle (Float64)
import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64

from ros2_parts.orientation import normalize_angle_rad, quaternion_to_yaw

R_EARTH_M = 6378137.0


def find_column(names, candidates):
    lower_map = {name.lower(): name for name in names}
    for candidate in candidates:
        if candidate in lower_map:
            return lower_map[candidate]
    raise ValueError(f"Could not find any of columns {candidates} in {names}")


class GpsPidNode(Node):
    def __init__(self):
        super().__init__("gps_pid")
        path_csv = self.declare_parameter("path_csv", "").value
        self.lookahead_m = self.declare_parameter("lookahead_m", 6.0).value
        self.throttle = self.declare_parameter("throttle", 0.22).value
        self.kp = self.declare_parameter("kp", 1.0).value
        self.ki = self.declare_parameter("ki", 0.0).value
        self.kd = self.declare_parameter("kd", 0.15).value
        self.max_steer_rad = math.radians(self.declare_parameter("max_steer_deg", 35.0).value)
        self.search_window = self.declare_parameter("search_window", 20).value

        # path in local XY, anchored at its first point (same datum gps_to_xy should use)
        data = np.genfromtxt(path_csv, delimiter=",", names=True, comments="#",
                             autostrip=True, dtype=float, encoding="utf-8")
        lat = np.radians(data[find_column(data.dtype.names, ("latitude", "lat", "lat_deg"))])
        lon = np.radians(data[find_column(data.dtype.names, ("longitude", "lon", "lon_deg"))])
        self.path_x_m = (lon - lon[0]) * math.cos(lat[0]) * R_EARTH_M
        self.path_y_m = (lat - lat[0]) * R_EARTH_M
        self.path_s_m = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(self.path_x_m), np.diff(self.path_y_m)))))

        self.last_closest_idx = None
        self.error_integral = 0.0
        self.last_error = 0.0
        self.last_time = None

        self.create_subscription(PoseStamped, "gps/pose", self.run, 10)
        self.steer_pub = self.create_publisher(Float64, "cmd/steering", 10)
        self.throt_pub = self.create_publisher(Float64, "cmd/throttle", 10)

    def find_closest_idx(self, x_m, y_m):
        # only search a window ahead, so the kart can't snap backwards where the path crosses itself
        if self.last_closest_idx is None:
            window = np.arange(len(self.path_x_m))
        else:
            window = np.arange(self.last_closest_idx, self.last_closest_idx + self.search_window + 1) % len(self.path_x_m)
        distances = np.hypot(self.path_x_m[window] - x_m, self.path_y_m[window] - y_m)
        self.last_closest_idx = int(window[np.argmin(distances)])
        return self.last_closest_idx

    def run(self, pose):
        x_m, y_m = pose.pose.position.x, pose.pose.position.y
        yaw_rad = quaternion_to_yaw(pose.pose.orientation)

        # lookahead point along the path, wrapping at the end
        closest_idx = self.find_closest_idx(x_m, y_m)
        target_s_m = self.path_s_m[closest_idx] + self.lookahead_m
        if target_s_m > self.path_s_m[-1]:
            target_s_m -= self.path_s_m[-1]
        target_idx = int(np.searchsorted(self.path_s_m, target_s_m, side="left"))

        desired_heading_rad = math.atan2(self.path_y_m[target_idx] - y_m, self.path_x_m[target_idx] - x_m)
        heading_error_rad = normalize_angle_rad(desired_heading_rad - yaw_rad)

        now = time.time()
        dt = 0.05 if self.last_time is None else max(now - self.last_time, 1e-3)
        self.last_time = now
        self.error_integral += heading_error_rad * dt
        derivative = (heading_error_rad - self.last_error) / dt
        self.last_error = heading_error_rad

        steering_angle_rad = self.kp * heading_error_rad + self.ki * self.error_integral + self.kd * derivative
        steering_cmd = float(np.clip(steering_angle_rad / self.max_steer_rad, -1.0, 1.0))

        self.steer_pub.publish(Float64(data=steering_cmd))
        self.throt_pub.publish(Float64(data=float(self.throttle)))


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
