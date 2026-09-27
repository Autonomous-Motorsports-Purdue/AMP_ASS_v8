#!/usr/bin/env python3
# in: gps/pose (PoseStamped), gps/vel (TwistStamped), imu/data (Imu), optional weights json | out: predicted_pose (PoseStamped)
import json
import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped, TwistStamped
from sensor_msgs.msg import Imu

from ros2_parts.orientation import quaternion_to_yaw, yaw_to_quaternion


class PredictiveModelNode(Node):
    def __init__(self):
        super().__init__("predictive_model")
        model_path = self.declare_parameter("model_path", "").value
        rate_hz = self.declare_parameter("rate_hz", 30.0).value

        # no weights = plain dead reckoning along the GPS course
        self.weights = None
        if model_path:
            with open(model_path, encoding="utf-8") as fh:
                self.weights = np.asarray(json.load(fh)["weights"], dtype=float)

        self.x = None
        self.y = None
        self.gps_yaw = None
        self.imu_yaw = None
        self.gps_speed = 0.0
        self.last_t = None

        self.create_subscription(PoseStamped, "gps/pose", self.on_pose, 10)
        self.create_subscription(TwistStamped, "gps/vel", self.on_vel, qos_profile_sensor_data)
        self.create_subscription(Imu, "imu/data", self.on_imu, qos_profile_sensor_data)
        self.pub = self.create_publisher(PoseStamped, "predicted_pose", 10)
        self.create_timer(1.0 / rate_hz, self.run)

    def on_pose(self, pose):
        self.x = pose.pose.position.x
        self.y = pose.pose.position.y
        self.gps_yaw = quaternion_to_yaw(pose.pose.orientation)

    def on_vel(self, vel):
        self.gps_speed = math.hypot(vel.twist.linear.x, vel.twist.linear.y)

    def on_imu(self, imu):
        self.imu_yaw = quaternion_to_yaw(imu.orientation)

    def run(self):
        if self.x is None:
            return
        now = self.get_clock().now().nanoseconds / 1e9
        if self.last_t is None:
            self.last_t = now
            return
        dt = max(now - self.last_t, 1e-4)
        self.last_t = now

        # features: speed * dt along the GPS course and along the IMU heading
        imu_yaw = self.imu_yaw if self.imu_yaw is not None else self.gps_yaw
        vdt = self.gps_speed * dt
        features = np.array([vdt * math.cos(self.gps_yaw), vdt * math.sin(self.gps_yaw),
                             vdt * math.cos(imu_yaw), vdt * math.sin(imu_yaw)])

        if self.weights is None:
            dx, dy = features[0], features[1]
        elif self.weights.shape[0] == 2:
            # one gps/imu blend shared by x and y
            w_gps, w_imu = self.weights[:, 0]
            dx = w_gps * features[0] + w_imu * features[2]
            dy = w_gps * features[1] + w_imu * features[3]
        else:
            dx, dy = features @ self.weights

        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = "map"
        pose.pose.position.x = float(self.x + dx)
        pose.pose.position.y = float(self.y + dy)
        pose.pose.orientation = yaw_to_quaternion(self.gps_yaw)
        self.pub.publish(pose)


def main(args=None):
    rclpy.init(args=args)
    node = PredictiveModelNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
