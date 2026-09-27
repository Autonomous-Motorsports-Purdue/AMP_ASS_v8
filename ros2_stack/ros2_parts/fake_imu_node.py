#!/usr/bin/env python3
# in: none | out: imu/data (Imu) -- body-frame accel and yaw while driving a circle
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu

from ros2_parts.orientation import yaw_to_quaternion


class FakeImuNode(Node):
    def __init__(self):
        super().__init__("fake_imu")
        imu_rate = self.declare_parameter("imu_rate", 100.0).value
        self.speed_mps = self.declare_parameter("speed_mps", 2.0).value
        turn_radius_m = self.declare_parameter("turn_radius_m", 8.0).value
        self.accel_noise_std = self.declare_parameter("accel_noise_std", 0.03).value
        seed = self.declare_parameter("seed", 42).value

        self.rng = np.random.default_rng(seed)
        self.imu_dt = 1.0 / imu_rate
        self.theta = 0.0
        self.yaw_rate = self.speed_mps / turn_radius_m

        self.pub = self.create_publisher(Imu, "imu/data", qos_profile_sensor_data)
        self.create_timer(self.imu_dt, self.run)

    def run(self):
        self.theta += self.yaw_rate * self.imu_dt

        # World-frame centripetal acceleration.
        ax_world = -self.speed_mps * self.yaw_rate * np.cos(self.theta)
        ay_world = -self.speed_mps * self.yaw_rate * np.sin(self.theta)
        heading_rad = self.theta + (np.pi / 2.0)

        # Rotate world acceleration into body frame.
        cos_h = np.cos(heading_rad)
        sin_h = np.sin(heading_rad)
        ax_body = ax_world * cos_h + ay_world * sin_h
        ay_body = -ax_world * sin_h + ay_world * cos_h

        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "imu_link"
        msg.orientation = yaw_to_quaternion(float(heading_rad))
        msg.angular_velocity.z = self.yaw_rate
        msg.linear_acceleration.x = float(ax_body + self.rng.normal(0.0, self.accel_noise_std))
        msg.linear_acceleration.y = float(ay_body + self.rng.normal(0.0, self.accel_noise_std))
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
