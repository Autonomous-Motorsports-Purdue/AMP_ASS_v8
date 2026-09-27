#!/usr/bin/env python3
# in: serial IMU lines (ox,oy,oz,gx,gy,gz,ax,ay,az) | out: imu/data (Imu)
import io
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
import serial

from ros2_parts.orientation import yaw_to_quaternion


class ImuNode(Node):
    def __init__(self):
        super().__init__("imu")
        port = self.declare_parameter("port", "/dev/ttyACM1").value
        baud = self.declare_parameter("baud", 115200).value
        rate_hz = self.declare_parameter("rate_hz", 30.0).value

        self.yaw_rate = 0.0
        self.yaw = 0.0
        self.ax = 0.0
        self.ay = 0.0

        self.ser = serial.Serial(port, baud, timeout=1)
        self.ser_io = io.TextIOWrapper(io.BufferedRWPair(self.ser, self.ser))

        self.pub = self.create_publisher(Imu, "imu/data", qos_profile_sensor_data)
        self.create_timer(1.0 / rate_hz, self.run)

    def run(self):
        line = self.ser_io.readline().strip()
        self.ser_io.readline()  # calibration line, unused
        if "Cal" in line or not line:
            return
        try:
            parts = [p.strip() for p in line.split(",")]
            parts = [p[3:] if ":" in p else p for p in parts]  # remove 'gx=', etc.
            ox, oy, oz, gx, gy, gz, ax, ay, az = [float(v) for v in parts]
            self.yaw_rate = gz
            self.yaw = math.radians(ox)
            self.ax, self.ay = ax, ay
        except ValueError:
            self.get_logger().warn(f"bad IMU line: {line}", throttle_duration_sec=5.0)

        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "imu_link"
        msg.orientation = yaw_to_quaternion(self.yaw)
        msg.angular_velocity.z = math.radians(self.yaw_rate)
        msg.linear_acceleration.x = float(self.ax)
        msg.linear_acceleration.y = float(self.ay)
        self.pub.publish(msg)

    def destroy_node(self):
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
