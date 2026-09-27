#!/usr/bin/env python3
# in: gps/fix, gps/vel, gps/fix_type, imu/data, odometry/filtered, cmd/*, controller/*, loop/*, video/* | out: data/information/<start>.csv, one row per tick
import csv
import datetime
import math
import os

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, NavSatFix
from std_msgs.msg import Float64, Int32, Int64, String

from ros2_parts.orientation import quaternion_to_yaw

FIELDS = [
    "timestamp", "latitude", "longitude", "steering", "throttle", "commanded_steer",
    "commanded_throttle", "cte", "idx", "fix", "gps_heading", "gps_speed", "imu_heading",
    "imu_accuracy_deg", "fused_x", "fused_y", "fused_yaw", "loop_index", "loop_monotonic_ns",
    "loop_wall_time", "video_nearest_camera_frame_id", "video_nearest_frame_pts_ns",
    "video_nearest_frame_monotonic_ns", "video_time_s", "video_delta_loop_to_frame_ms", "video_path",
]


class LoggerGpsNode(Node):
    def __init__(self):
        super().__init__("logger_gps")
        rate_hz = self.declare_parameter("rate_hz", 50.0).value

        os.makedirs("data/information", exist_ok=True)
        start_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.csvfile = open(f"data/information/{start_time}.csv", "w", newline="")
        self.csvwriter = csv.DictWriter(self.csvfile, fieldnames=FIELDS)
        self.csvwriter.writeheader()

        # latest value of every column; columns nothing publishes stay empty
        self.row = {field: None for field in FIELDS}

        self.create_subscription(NavSatFix, "gps/fix", self.on_fix, qos_profile_sensor_data)
        self.create_subscription(TwistStamped, "gps/vel", self.on_vel, qos_profile_sensor_data)
        self.create_subscription(Imu, "imu/data", self.on_imu, qos_profile_sensor_data)
        self.create_subscription(Odometry, "odometry/filtered", self.on_odometry, 10)
        for msg_type, topic, field in (
            (String, "gps/fix_type", "fix"),
            (Float64, "cmd/steering", "steering"),
            (Float64, "cmd/throttle", "throttle"),
            (Float64, "cmd/commanded_steering", "commanded_steer"),
            (Float64, "cmd/commanded_throttle", "commanded_throttle"),
            (Float64, "controller/cross_track_error", "cte"),
            (Int32, "controller/waypoint_index", "idx"),
            (Int64, "loop/index", "loop_index"),
            (Int64, "loop/monotonic_ns", "loop_monotonic_ns"),
            (Int64, "video/frame_id", "video_nearest_camera_frame_id"),
            (Float64, "video/time_s", "video_time_s"),
            (Float64, "video/delta_ms", "video_delta_loop_to_frame_ms"),
            (String, "video/path", "video_path"),
        ):
            self.create_subscription(msg_type, topic, lambda msg, f=field: self.row.__setitem__(f, msg.data), 10)

        self.create_timer(1.0 / rate_hz, self.run)

    def on_fix(self, fix):
        self.row["latitude"] = fix.latitude
        self.row["longitude"] = fix.longitude

    def on_vel(self, vel):
        east, north = vel.twist.linear.x, vel.twist.linear.y
        self.row["gps_speed"] = math.hypot(east, north)
        self.row["gps_heading"] = math.degrees(math.atan2(north, east))

    def on_imu(self, imu):
        self.row["imu_heading"] = math.degrees(quaternion_to_yaw(imu.orientation))
        cov = imu.orientation_covariance
        self.row["imu_accuracy_deg"] = None if cov[0] < 0.0 else math.degrees(math.sqrt(max(cov[8], 0.0)))

    def on_odometry(self, odom):
        self.row["fused_x"] = odom.pose.pose.position.x
        self.row["fused_y"] = odom.pose.pose.position.y
        self.row["fused_yaw"] = math.degrees(quaternion_to_yaw(odom.pose.pose.orientation))

    def run(self):
        self.row["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%f")
        self.csvwriter.writerow(self.row)
        self.csvfile.flush()

    def destroy_node(self):
        self.csvfile.close()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LoggerGpsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
