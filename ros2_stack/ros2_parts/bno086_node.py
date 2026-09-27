#!/usr/bin/env python3
# in: BNO086 serial key=value lines | out: imu/data (Imu, ENU yaw with declination applied, yaw covariance from accuracy_deg)
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

KV_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^,\s]+)")


def field_float(fields, name, default=None):
    try:
        out = float(fields[name])
    except (KeyError, ValueError):
        return default
    return out if math.isfinite(out) else default


class Bno086Node(Node):
    def __init__(self):
        super().__init__("bno086")
        port = self.declare_parameter("port", "/dev/ttyACM2").value
        baudrate = self.declare_parameter("baudrate", 460800).value
        self.declination = self.declare_parameter("declination", -4.55).value
        self.mount_offset = self.declare_parameter("mount_offset", 0.0).value
        self.invert = self.declare_parameter("invert", False).value
        rate_hz = self.declare_parameter("rate_hz", 30.0).value

        self.ser = serial.Serial(port, baudrate, timeout=0.05)
        self.latest = None

        self.pub = self.create_publisher(Imu, "imu/data", qos_profile_sensor_data)
        self.create_timer(1.0 / rate_hz, self.run)

    def run(self):
        # drain the port so we always publish the newest sample
        while self.ser.in_waiting:
            line = self.ser.readline().decode(errors="ignore").strip()
            sample = self.parse_line(line)
            if sample is not None:
                self.latest = sample
        if self.latest is not None:
            self.pub.publish(self.latest)

    def parse_line(self, line):
        fields = dict(KV_RE.findall(line))
        q = [field_float(fields, k) for k in ("qi", "qj", "qk", "qr")]
        if None in q:
            return None
        quat = normalize_quaternion(*q)
        if quat is None:
            return None

        x, y, z, w = quat
        if self.invert:
            x, y, z = -x, -y, -z

        roll, pitch, yaw = quaternion_to_euler_deg(x, y, z, w)
        heading = wrap360(enu_yaw_to_compass_heading_deg(yaw) + self.mount_offset + self.declination)

        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "imu_link"
        msg.orientation = yaw_to_quaternion(math.radians(compass_heading_to_enu_yaw_deg(heading)))

        accuracy_deg = field_float(fields, "accuracy_deg")
        if accuracy_deg is None:
            msg.orientation_covariance[0] = -1.0  # -1 means unknown
        else:
            msg.orientation_covariance[8] = math.radians(accuracy_deg) ** 2  # yaw variance

        msg.angular_velocity.x = math.radians(field_float(fields, "gx_dps", 0.0))
        msg.angular_velocity.y = math.radians(field_float(fields, "gy_dps", 0.0))
        msg.angular_velocity.z = math.radians(field_float(fields, "gz_dps", 0.0))
        msg.linear_acceleration.x = field_float(fields, "lin_ax_mps2", 0.0)
        msg.linear_acceleration.y = field_float(fields, "lin_ay_mps2", 0.0)
        msg.linear_acceleration.z = field_float(fields, "lin_az_mps2", 0.0)
        return msg

    def destroy_node(self):
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
