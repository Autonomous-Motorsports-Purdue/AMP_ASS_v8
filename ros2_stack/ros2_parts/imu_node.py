#!/usr/bin/env python3
"""ROS 2 port of parts/imu.py.

Serial IMU with 9-value output: ox, oy, oz, gx, gy, gz, ax, ay, az.
gz = yaw rate (deg/s). Only the channels the part kept are filled in: yaw,
the z rotation rate and the x/y accelerations. The rest are left zero.

The board reports its yaw rate in deg/s and the part passed that through
as-is. sensor_msgs/Imu is specified in rad/s, so it is converted here.
Anything tuned against the old value, the closed-loop yaw-rate controller in
particular, needs its gains rechecked.

publishes: imu/data (Imu)
"""

import io
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
import serial

from ros2_parts.orientation import yaw_to_quaternion
from ros2_parts.parameters import declare


class ImuNode(Node):
    def __init__(self):
        super().__init__("imu")

        self.port = declare(self, "port", "/dev/ttyACM1", "Serial port the IMU is on.")
        self.baud = declare(self, "baud", 115200, "Serial baud rate.")
        self.frame_id = declare(
            self, "frame_id", "imu_link", "Frame the measurements are expressed in.")
        self.debug = declare(self, "debug", False, "Log parse errors.")
        rate_hz = declare(self, "rate_hz", 30.0, "Rate the port is polled at, in Hz.")

        # latched, so a dropped or unparsable line republishes the last good
        # sample rather than a zero, as the part did
        self.yaw_rate = 0.0
        self.yaw = 0.0
        self.ax = 0.0
        self.ay = 0.0

        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=1)
            self.ser_io = io.TextIOWrapper(io.BufferedRWPair(self.ser, self.ser))
            self.get_logger().info(f"Connected to {self.port} @ {self.baud}")
        except Exception as e:
            self.get_logger().error(f"Could not open serial port: {e}")
            self.ser = None
            self.ser_io = None

        self.pub = self.create_publisher(Imu, "imu/data", qos_profile_sensor_data)
        self.create_timer(1.0 / rate_hz, self.publish_sample)

    def run(self):
        """Read a line of 9 comma-separated IMU values and return yaw_rate."""
        if not self.ser_io:
            self.get_logger().error("IMU ERROR", throttle_duration_sec=5.0)
            return self.yaw_rate, self.yaw, 0, 0

        line = self.ser_io.readline().strip()
        cal = self.ser_io.readline().strip()
        if "Cal" in line:
            self.get_logger().debug("Recieved Calibration Packet, ret 0's")
            return 0, 0, 0, 0
        if not line:
            self.get_logger().warn("Error reading from IMU", throttle_duration_sec=5.0)
            return self.yaw_rate, self.yaw, 0, 0
        try:
            parts = [p.strip() for p in line.split(",")]
            parts = [p[3:] if ":" in p else p for p in parts]  # remove 'gx=', etc.
            ox, oy, oz, gx, gy, gz, ax, ay, az = [float(v) for v in parts]

            self.yaw_rate = gz
            self.yaw = math.radians(ox)

            self.ax, self.ay = ax, ay

        except Exception as e:
            if self.debug:
                self.get_logger().warn(f"Parse error: {e} | Line: {line}")

        return self.yaw_rate, self.yaw, self.ax, self.ay

    def publish_sample(self):
        yaw_rate, yaw, ax, ay = self.run()

        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.orientation = yaw_to_quaternion(yaw)
        msg.angular_velocity.z = math.radians(yaw_rate)
        msg.linear_acceleration.x = float(ax)
        msg.linear_acceleration.y = float(ay)
        self.pub.publish(msg)

    def destroy_node(self):
        if self.ser is not None and self.ser.is_open:
            self.ser.close()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ImuNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
