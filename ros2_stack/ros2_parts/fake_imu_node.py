#!/usr/bin/env python3
"""ROS 2 port of parts/fake_imu.py.

Simulates body-frame IMU acceleration and yaw for bench testing, so it stands
in for imu_node and bno086_node. Motion is a constant-speed circle to produce
non-zero dynamics.

publishes: imu/data (Imu)
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu

from ros2_parts.orientation import yaw_to_quaternion
from ros2_parts.parameters import declare


class FakeImuNode(Node):
    def __init__(self):
        super().__init__("fake_imu")

        imu_rate = declare(self, "imu_rate", 100.0, "Simulated sample rate, in Hz.")
        self.speed_mps = declare(self, "speed_mps", 2.0, "Simulated ground speed.")
        self.turn_radius_m = declare(
            self, "turn_radius_m", 8.0, "Radius of the circle driven, in meters.")
        self.accel_noise_std = declare(
            self, "accel_noise_std", 0.03, "Std dev of the acceleration noise, in m/s^2.")
        seed = declare(self, "seed", 42, "Seed for the noise generator.")
        self.frame_id = declare(
            self, "frame_id", "imu_link", "Frame the measurements are expressed in.")

        self.rng = np.random.default_rng(seed)
        self.imu_dt = 1.0 / imu_rate
        self.theta = 0.0
        self.yaw_rate = self.speed_mps / self.turn_radius_m

        self.pub = self.create_publisher(Imu, "imu/data", qos_profile_sensor_data)
        self.create_timer(self.imu_dt, self.publish_sample)

    def run(self):
        self.theta += self.yaw_rate * self.imu_dt

        # World-frame centripetal acceleration.
        ax_world = -self.speed_mps * self.yaw_rate * np.cos(self.theta)
        ay_world = -self.speed_mps * self.yaw_rate * np.sin(self.theta)
        # heading leads the position angle by a quarter turn on a circle
        heading_rad = self.theta + (np.pi / 2.0)

        # Rotate world acceleration into body frame.
        cos_h = np.cos(heading_rad)
        sin_h = np.sin(heading_rad)
        ax_body = ax_world * cos_h + ay_world * sin_h
        ay_body = -ax_world * sin_h + ay_world * cos_h

        ax_meas = ax_body + self.rng.normal(0.0, self.accel_noise_std)
        ay_meas = ay_body + self.rng.normal(0.0, self.accel_noise_std)

        return float(ax_meas), float(ay_meas), float(heading_rad)

    def publish_sample(self):
        accel_x, accel_y, heading_rad = self.run()

        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.orientation = yaw_to_quaternion(heading_rad)
        msg.angular_velocity.z = self.yaw_rate
        msg.linear_acceleration.x = accel_x
        msg.linear_acceleration.y = accel_y
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = FakeImuNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
