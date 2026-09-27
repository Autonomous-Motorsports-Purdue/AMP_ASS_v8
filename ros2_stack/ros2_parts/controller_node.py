#!/usr/bin/env python3
# mpc_part -- in: gps/pose (PoseStamped), path_csv (x, y, psi_rad) | out: cmd/desired_yaw_rate, cmd/desired_speed (Float64)
# closed_loop_controller -- in: cmd/desired_yaw_rate, cmd/desired_speed (Float64), imu/data (Imu) | out: cmd/steering, cmd/throttle (Float64)
import math
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64

from ros2_parts.orientation import quaternion_to_yaw


class MpcPartNode(Node):
    def __init__(self):
        super().__init__("mpc_part")
        path_csv = self.declare_parameter("path_csv", "").value
        self.horizon = self.declare_parameter("horizon", 2).value
        self.dt_mpc = self.declare_parameter("dt_mpc", 0.1).value
        self.wheelbase = self.declare_parameter("wheelbase", 1.000506).value
        self.max_steer = math.radians(self.declare_parameter("max_steer_deg", 35.0).value)
        self.target_speed = self.declare_parameter("target_speed", 2.0).value
        self.k_y = 1.0
        self.k_smooth = 0.25

        df = pd.read_csv(path_csv)
        df.columns = df.columns.str.strip()
        df.rename(columns={"x_m": "x", "y_m": "y"}, inplace=True)
        self.path = df[["x", "y", "psi_rad"]].to_numpy()

        self.create_subscription(PoseStamped, "gps/pose", self.run, 10)
        self.yaw_rate_pub = self.create_publisher(Float64, "cmd/desired_yaw_rate", 10)
        self.speed_pub = self.create_publisher(Float64, "cmd/desired_speed", 10)

    def cost_function(self, steering_sequence, x, y, yaw, closest_idx):
        # roll a kinematic bicycle forward and score distance to the path plus steering changes
        cost = 0.0
        idx = closest_idx
        prev_steer = 0.0
        for steer in np.clip(steering_sequence, -self.max_steer, self.max_steer):
            yaw += (self.target_speed / self.wheelbase) * math.tan(steer) * self.dt_mpc
            x += self.target_speed * math.cos(yaw) * self.dt_mpc
            y += self.target_speed * math.sin(yaw) * self.dt_mpc
            idx = min(idx + 1, len(self.path) - 1)
            cost += self.k_y * ((self.path[idx, 0] - x) ** 2 + (self.path[idx, 1] - y) ** 2)
            cost += self.k_smooth * (steer - prev_steer) ** 2
            prev_steer = steer
        return cost

    def run(self, pose):
        x, y = pose.pose.position.x, pose.pose.position.y
        yaw = quaternion_to_yaw(pose.pose.orientation)
        closest_idx = int(np.argmin(np.hypot(self.path[:, 0] - x, self.path[:, 1] - y)))

        # receding horizon: solve the whole sequence, use only the first steer
        res = minimize(self.cost_function, np.zeros(self.horizon), args=(x, y, yaw, closest_idx),
                       bounds=[(-self.max_steer, self.max_steer)] * self.horizon, method="SLSQP",
                       options={"ftol": 1e-3, "disp": False, "maxiter": 30})
        desired_ang_vel = (self.target_speed / self.wheelbase) * math.tan(res.x[0])

        self.yaw_rate_pub.publish(Float64(data=float(desired_ang_vel)))
        self.speed_pub.publish(Float64(data=float(self.target_speed)))


class ClosedLoopControllerNode(Node):
    def __init__(self):
        super().__init__("closed_loop_controller")
        self.kp = self.declare_parameter("kp", 0.2).value
        self.ki = self.declare_parameter("ki", 0.0).value
        self.kd = self.declare_parameter("kd", 0.0).value
        self.max_steering_deg = self.declare_parameter("max_steering_deg", 35.0).value
        rate_hz = self.declare_parameter("rate_hz", 30.0).value
        self.integral_limit = 10.0

        self.error_integral = 0.0
        self.last_error = 0.0
        self.last_time = None

        self.desired_ang_vel = None
        self.actual_ang_vel = None
        self.target_speed = None

        self.create_subscription(Float64, "cmd/desired_yaw_rate", self.on_desired_yaw_rate, 10)
        self.create_subscription(Float64, "cmd/desired_speed", self.on_desired_speed, 10)
        self.create_subscription(Imu, "imu/data", self.on_imu, qos_profile_sensor_data)
        self.steer_pub = self.create_publisher(Float64, "cmd/steering", 10)
        self.throt_pub = self.create_publisher(Float64, "cmd/throttle", 10)
        self.create_timer(1.0 / rate_hz, self.run)

    def on_desired_yaw_rate(self, msg):
        self.desired_ang_vel = msg.data

    def on_desired_speed(self, msg):
        self.target_speed = msg.data

    def on_imu(self, msg):
        self.actual_ang_vel = msg.angular_velocity.z

    def run(self):
        if self.desired_ang_vel is None or self.actual_ang_vel is None or self.target_speed is None:
            return
        error = self.desired_ang_vel - self.actual_ang_vel

        now = time.time()
        dt = 0.01 if self.last_time is None else max(now - self.last_time, 0.001)
        self.last_time = now

        self.error_integral = np.clip(self.error_integral + error * dt, -self.integral_limit, self.integral_limit)
        steering_rad = self.kp * error + self.ki * self.error_integral + self.kd * (error - self.last_error) / dt
        self.last_error = error

        steering_deg = np.clip(math.degrees(steering_rad), -self.max_steering_deg, self.max_steering_deg)
        # invert: positive yaw rate turns left, positive steering turns right
        steering_value = -float(np.clip(steering_deg / self.max_steering_deg, -1.0, 1.0))

        self.steer_pub.publish(Float64(data=steering_value))
        self.throt_pub.publish(Float64(data=float(self.target_speed)))


def main(args=None):
    rclpy.init(args=args)
    node = MpcPartNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


def main_closed_loop(args=None):
    rclpy.init(args=args)
    node = ClosedLoopControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
