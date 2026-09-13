#!/usr/bin/env python3
"""ROS 2 port of parts/bno086.py.

The board's heading is compass style: zero at north, increasing clockwise.
ROS orientation is ENU yaw: zero at east, increasing counter-clockwise. The
declination and mounting offsets are applied in compass degrees, exactly as
the part did, and the result is converted once on the way out.

The board reports its own heading accuracy. That goes in the yaw term of
orientation_covariance, which is where a ROS consumer looks for it, so there
is no separate accuracy topic.

publishes: imu/data (Imu)
"""

import math
import re

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
import serial

from ros2_parts.orientation import (
    compass_heading_to_enu_yaw_deg,
    enu_yaw_to_compass_heading_deg,
    normalize_quaternion,
    quaternion_to_euler_deg,
    wrap360,
    yaw_to_quaternion,
)
from ros2_parts.parameters import declare

KV_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^,\s]+)")

# yaw variance sits at index 8 of a row-major 3x3 orientation covariance
YAW_COVARIANCE_INDEX = 8


def parse_float(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def field_float(fields, name, default=None):
    value = parse_float(fields.get(name))
    return default if value is None else value


class Bno086Node(Node):
    def __init__(self):
        super().__init__("bno086")

        port = declare(self, "port", "/dev/ttyACM2", "Serial port the IMU is on.")
        baudrate = declare(self, "baudrate", 460800, "Serial baud rate.")
        timeout = declare(self, "timeout", 0.05, "Serial read timeout, in seconds.")
        poll_delay = declare(self, "poll_delay", 0.0025, "How often the port is drained.")
        self.declination = declare(self, "declination", -4.55, "Magnetic declination, in degrees.")
        self.mount_offset = declare(
            self, "mount_offset", 0.0, "Heading offset for the board mounting, in degrees.")
        self.invert = declare(self, "invert", False, "Invert the quaternion vector part.")
        self.frame_id = declare(
            self, "frame_id", "imu_link", "Frame the measurements are expressed in.")
        rate_hz = declare(self, "rate_hz", 30.0, "Rate samples are published at, in Hz.")

        self.ser = serial.Serial(port, baudrate, timeout=timeout)
        self.get_logger().info(f"Connected to {port} @ {baudrate}")

        self.latest = None

        self.pub = self.create_publisher(Imu, "imu/data", qos_profile_sensor_data)
        # drained faster than it is published, so each published sample is the
        # most recent one the board sent rather than a backlogged one
        self.create_timer(poll_delay, self.run)
        self.create_timer(1.0 / rate_hz, self.publish_latest)

    def run(self):
        while self.ser.in_waiting:
            line = self.ser.readline().decode(errors="ignore").strip()
            sample = self._parse_line(line)
            if sample is not None:
                self.latest = sample

        return self.latest

    def _parse_line(self, line):
        fields = dict(KV_RE.findall(line))
        if not {"qi", "qj", "qk", "qr"}.issubset(fields):
            return None

        qi = parse_float(fields["qi"])
        qj = parse_float(fields["qj"])
        qk = parse_float(fields["qk"])
        qr = parse_float(fields["qr"])
        if None in (qi, qj, qk, qr):
            return None

        quat = normalize_quaternion(qi, qj, qk, qr)
        if quat is None:
            return None

        x, y, z, w = quat
        if self.invert:
            x, y, z = -x, -y, -z

        roll, pitch, yaw = quaternion_to_euler_deg(x, y, z, w)
        heading = wrap360(
            enu_yaw_to_compass_heading_deg(yaw) + self.mount_offset + self.declination)

        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.orientation = yaw_to_quaternion(
            math.radians(compass_heading_to_enu_yaw_deg(heading)))

        accuracy_deg = field_float(fields, "accuracy_deg")
        if accuracy_deg is None:
            # per the message spec, a negative leading element means unknown
            # rather than zero
            msg.orientation_covariance[0] = -1.0
        else:
            msg.orientation_covariance[YAW_COVARIANCE_INDEX] = math.radians(accuracy_deg) ** 2

        msg.angular_velocity.x = math.radians(field_float(fields, "gx_dps", 0.0))
        msg.angular_velocity.y = math.radians(field_float(fields, "gy_dps", 0.0))
        msg.angular_velocity.z = math.radians(field_float(fields, "gz_dps", 0.0))

        msg.linear_acceleration.x = field_float(fields, "lin_ax_mps2", 0.0)
        msg.linear_acceleration.y = field_float(fields, "lin_ay_mps2", 0.0)
        msg.linear_acceleration.z = field_float(fields, "lin_az_mps2", 0.0)

        return msg

    def publish_latest(self):
        if self.latest is None:
            self.get_logger().warn("no valid sample yet", throttle_duration_sec=5.0)
            return
        self.pub.publish(self.latest)

    def destroy_node(self):
        if self.ser is not None and self.ser.is_open:
            self.ser.close()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = Bno086Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
