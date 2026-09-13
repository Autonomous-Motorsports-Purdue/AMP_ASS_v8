#!/usr/bin/env python3
"""ROS 2 port of parts/logger_gps.py.

Logs a whole GPS driving run to a timestamped CSV. The columns are unchanged,
so the existing plotting and analysis scripts still read the output. A row is
written on a timer holding the most recent value of each column, which is what
the part did: one row per drive loop, each column whatever was in memory at
the time. Columns with no topic behind them stay empty.

subscribes: gps/fix, gps/vel, gps/fix_type, imu/data, odometry/filtered,
            the cmd/ and controller/ scalars, loop/ and video/
"""

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
from ros2_parts.parameters import declare

# yaw variance sits at index 8 of a row-major 3x3 orientation covariance
YAW_COVARIANCE_INDEX = 8


class LoggerGpsNode(Node):
    def __init__(self):
        super().__init__("logger_gps")

        data_dir = declare(
            self, "data_dir", "data/information", "Directory the CSV is written to.")
        rate_hz = declare(self, "rate_hz", 50.0, "Rate rows are written at, in Hz.")

        start_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.information_directory = data_dir
        if not os.path.exists(self.information_directory):
            os.makedirs(self.information_directory)
        self.info_csv = os.path.join(self.information_directory, start_time + ".csv")
        # Writing to csv file
        self.csvfile = open(self.info_csv, "w", newline="")

        self.base_fields = [
            "timestamp",
            "latitude",
            "longitude",
            "steering",
            "throttle",
            "commanded_steer",
            "commanded_throttle",
            "cte",
            "idx",
            "fix",
            "gps_heading",
            "gps_speed",
            "imu_heading",
            "imu_accuracy_deg",
            "fused_x",
            "fused_y",
            "fused_yaw",
            "loop_index",
            "loop_monotonic_ns",
            "loop_wall_time",
            "video_nearest_camera_frame_id",
            "video_nearest_frame_pts_ns",
            "video_nearest_frame_monotonic_ns",
            "video_time_s",
            "video_delta_loop_to_frame_ms",
            "video_path",
        ]
        self.fields = self.base_fields
        self.csvwriter = csv.DictWriter(self.csvfile, fieldnames=self.fields)

        # Writing the fields
        self.csvwriter.writeheader()
        self.get_logger().info(f"Logging the run to {self.info_csv}")

        self.row = {field: None for field in self.fields}

        self.create_subscription(NavSatFix, "gps/fix", self.on_fix, qos_profile_sensor_data)
        self.create_subscription(
            TwistStamped, "gps/vel", self.on_velocity, qos_profile_sensor_data)
        self.create_subscription(String, "gps/fix_type", self.store("fix"), 10)
        self.create_subscription(Imu, "imu/data", self.on_imu, qos_profile_sensor_data)
        self.create_subscription(Odometry, "odometry/filtered", self.on_odometry, 10)
        self.create_subscription(Float64, "cmd/steering", self.store("steering"), 10)
        self.create_subscription(Float64, "cmd/throttle", self.store("throttle"), 10)
        self.create_subscription(
            Float64, "cmd/commanded_steering", self.store("commanded_steer"), 10)
        self.create_subscription(
            Float64, "cmd/commanded_throttle", self.store("commanded_throttle"), 10)
        self.create_subscription(Float64, "controller/cross_track_error", self.store("cte"), 10)
        self.create_subscription(Int32, "controller/waypoint_index", self.store("idx"), 10)
        self.create_subscription(Int64, "loop/index", self.store("loop_index"), 10)
        self.create_subscription(Int64, "loop/monotonic_ns", self.store("loop_monotonic_ns"), 10)
        self.create_subscription(
            Int64, "video/frame_id", self.store("video_nearest_camera_frame_id"), 10)
        self.create_subscription(Float64, "video/time_s", self.store("video_time_s"), 10)
        self.create_subscription(
            Float64, "video/delta_ms", self.store("video_delta_loop_to_frame_ms"), 10)
        self.create_subscription(String, "video/path", self.store("video_path"), 10)

        self.create_timer(1.0 / rate_hz, self.run)

    def store(self, field):
        """Return a callback that files a message's value under field."""
        def callback(msg):
            self.row[field] = msg.data
        return callback

    def on_fix(self, fix):
        self.row["latitude"] = fix.latitude
        self.row["longitude"] = fix.longitude

    def on_velocity(self, vel):
        east = vel.twist.linear.x
        north = vel.twist.linear.y
        self.row["gps_speed"] = math.hypot(east, north)
        self.row["gps_heading"] = math.degrees(math.atan2(north, east))

    def on_imu(self, imu):
        self.row["imu_heading"] = math.degrees(quaternion_to_yaw(imu.orientation))
        cov = imu.orientation_covariance
        if cov[0] < 0.0:
            self.row["imu_accuracy_deg"] = None
        else:
            self.row["imu_accuracy_deg"] = math.degrees(
                math.sqrt(max(cov[YAW_COVARIANCE_INDEX], 0.0)))

    def on_odometry(self, odom):
        self.row["fused_x"] = odom.pose.pose.position.x
        self.row["fused_y"] = odom.pose.pose.position.y
        self.row["fused_yaw"] = math.degrees(quaternion_to_yaw(odom.pose.pose.orientation))

    def run(self):
        """Write the current value of every column."""
        self.row["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%f")
        self.csvwriter.writerow(self.row)
        self.csvfile.flush()

    def destroy_node(self):
        if self.csvfile is not None and not self.csvfile.closed:
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
