#!/usr/bin/env python3
"""ROS 2 port of parts/predictive_model.py.

Predict next (x, y) from current position and GPS+IMU motion features.

Baseline (no trained weights):
  x_next = x + speed * cos(gps_yaw) * dt
  y_next = y + speed * sin(gps_yaw) * dt

Trained mode learns weights on the four v*dt*cos/sin terms. A 2-row weights
file shares one weight pair across both axes, a 4-row file gives each feature
its own. predict_step is the TinyPredictiveModel the cte_controller_node uses.

subscribes: gps/pose (PoseStamped), gps/vel (TwistStamped), imu/data (Imu)
publishes:  predicted_pose (PoseStamped)
"""

import json
import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped, TwistStamped
from sensor_msgs.msg import Imu

from ros2_parts.orientation import quaternion_to_yaw, yaw_to_quaternion
from ros2_parts.parameters import declare

FEATURE_NAMES = [
    "vdt_cos_gps_yaw",
    "vdt_sin_gps_yaw",
    "vdt_cos_imu_yaw",
    "vdt_sin_imu_yaw",
]


def predict_step(x, y, gps_speed, gps_yaw_rad, dt):
    """One step of dead reckoning. Returns the input unchanged if anything
    is missing. This is TinyPredictiveModel.run, in radians."""
    inputs = (x, y, gps_speed, gps_yaw_rad)
    if any([i is None for i in inputs]):
        return x, y
    else:
        x_f = x + gps_speed * math.cos(gps_yaw_rad) * dt
        y_f = y + gps_speed * math.sin(gps_yaw_rad) * dt
        return x_f, y_f


def build_step_features(gps_speed_mps, dt, gps_yaw_rad, imu_yaw_rad):
    """
    Interpretable one-step motion features:
      delta_x ~= sum(w_i * feature_i)
      delta_y ~= sum(w_j * feature_j)

    Each feature is speed * direction_component * dt.
    """
    dt = max(float(dt), 1e-4)
    speed = float(gps_speed_mps or 0.0)

    return np.array(
        [
            speed * math.cos(gps_yaw_rad) * dt,
            speed * math.sin(gps_yaw_rad) * dt,
            speed * math.cos(imu_yaw_rad) * dt,
            speed * math.sin(imu_yaw_rad) * dt,
        ],
        dtype=float,
    )


class PredictiveModelNode(Node):
    def __init__(self):
        super().__init__("predictive_model")

        model_path = declare(
            self, "model_path", "", "JSON of trained weights; empty uses dead reckoning.")
        self.frame_id = declare(
            self, "frame_id", "map", "Frame the predicted pose is expressed in.")
        rate_hz = declare(self, "rate_hz", 30.0, "Rate predictions are published at, in Hz.")

        self.weights = self.load(model_path)

        self.x = None
        self.y = None
        self.gps_yaw_rad = None
        self.imu_yaw_rad = None
        self.gps_speed = None
        self.timer = None

        self.pub = self.create_publisher(PoseStamped, "predicted_pose", 10)
        self.create_subscription(PoseStamped, "gps/pose", self.on_pose, 10)
        self.create_subscription(
            TwistStamped, "gps/vel", self.on_velocity, qos_profile_sensor_data)
        self.create_subscription(Imu, "imu/data", self.on_imu, qos_profile_sensor_data)
        self.create_timer(1.0 / rate_hz, self.publish_prediction)

    def load(self, path):
        if not path:
            self.get_logger().info("no weights given, predicting by dead reckoning")
            return None

        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        weights = np.asarray(payload["weights"], dtype=float)
        self.get_logger().info(
            f"Loaded {weights.shape[0]} weights from {path} "
            f"over {payload.get('feature_names', FEATURE_NAMES)}")
        return weights

    def on_pose(self, pose):
        self.x = pose.pose.position.x
        self.y = pose.pose.position.y
        self.gps_yaw_rad = quaternion_to_yaw(pose.pose.orientation)

    def on_velocity(self, vel):
        self.gps_speed = math.hypot(vel.twist.linear.x, vel.twist.linear.y)

    def on_imu(self, imu):
        self.imu_yaw_rad = quaternion_to_yaw(imu.orientation)

    def _predict_delta(self, gps_speed, gps_yaw_rad, imu_yaw_rad, dt=0.02):
        features = build_step_features(gps_speed, dt, gps_yaw_rad, imu_yaw_rad)
        if self.weights is None:
            return float(features[0]), float(features[1])

        if self.weights.shape[0] == 2:
            # Shared GPS/IMU blend: same two weights for x and y.
            w_gps, w_imu = self.weights[:, 0]
            dx = w_gps * features[0] + w_imu * features[2]
            dy = w_gps * features[1] + w_imu * features[3]
            return float(dx), float(dy)

        delta = features @ self.weights
        return float(delta[0]), float(delta[1])

    def run(self, x_t, y_t, gps_speed, gps_yaw_rad, imu_yaw_rad, dt):
        inputs = (x_t, y_t, gps_speed)
        if any(v is not None and not np.isfinite(float(v)) for v in inputs):
            return x_t, y_t

        if gps_yaw_rad is None:
            return x_t, y_t
        if imu_yaw_rad is None:
            imu_yaw_rad = gps_yaw_rad

        dx, dy = self._predict_delta(gps_speed, gps_yaw_rad, imu_yaw_rad, dt=dt)
        return float(x_t) + dx, float(y_t) + dy

    def publish_prediction(self):
        if self.x is None or self.y is None or self.gps_yaw_rad is None:
            return

        now = self.get_clock().now().nanoseconds / 1e9
        if self.timer is None:
            self.timer = now
            return
        dt = now - self.timer
        self.timer = now

        x_f, y_f = self.run(
            self.x, self.y, self.gps_speed, self.gps_yaw_rad, self.imu_yaw_rad, dt)

        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = self.frame_id
        pose.pose.position.x = x_f
        pose.pose.position.y = y_f
        pose.pose.orientation = yaw_to_quaternion(self.gps_yaw_rad)
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
