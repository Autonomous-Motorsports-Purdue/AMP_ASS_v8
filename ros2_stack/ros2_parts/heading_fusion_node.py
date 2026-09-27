#!/usr/bin/env python3
# in: gps/pose (PoseStamped), gps/vel (TwistStamped), imu/data (Imu) | out: odometry/filtered (Odometry) from a 6-state EKF [x, y, vx, vy, yaw, gyro_bias]
import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu

from ros2_parts.orientation import normalize_angle_rad, quaternion_to_yaw, yaw_to_quaternion

GPS_POSITION_VAR = 0.10 ** 2
GPS_VELOCITY_VAR = 0.10 ** 2
GPS_YAW_VAR = math.radians(5.0) ** 2
IMU_YAW_VAR = math.radians(30.0) ** 2
ACCEL_VAR = 2.0 ** 2
GYRO_VAR = math.radians(1.5) ** 2
GYRO_BIAS_VAR = math.radians(0.1) ** 2


class HeadingFusionNode(Node):
    def __init__(self):
        super().__init__("heading_fusion")
        self.min_gps_speed_mps = self.declare_parameter("min_gps_speed_mps", 1.0).value
        self.max_imu_accuracy_deg = self.declare_parameter("max_imu_accuracy_deg", 15.0).value
        rate_hz = self.declare_parameter("rate_hz", 50.0).value

        self.state = None
        self.cov = None
        self.last_t = None

        self.gps_x = None
        self.gps_y = None
        self.gps_yaw = None
        self.gps_speed = None
        self.imu = None

        self.create_subscription(PoseStamped, "gps/pose", self.on_gps_pose, 10)
        self.create_subscription(TwistStamped, "gps/vel", self.on_gps_vel, qos_profile_sensor_data)
        self.create_subscription(Imu, "imu/data", self.on_imu, qos_profile_sensor_data)
        self.pub = self.create_publisher(Odometry, "odometry/filtered", 10)
        self.create_timer(1.0 / rate_hz, self.run)

    def on_gps_pose(self, pose):
        self.gps_x = pose.pose.position.x
        self.gps_y = pose.pose.position.y
        self.gps_yaw = quaternion_to_yaw(pose.pose.orientation)

    def on_gps_vel(self, vel):
        self.gps_speed = math.hypot(vel.twist.linear.x, vel.twist.linear.y)

    def on_imu(self, imu):
        self.imu = imu

    def imu_yaw(self):
        # skip the IMU heading when it reports poor accuracy (cov[0] < 0 means unknown, which is fine)
        if self.imu is None:
            return None
        cov = self.imu.orientation_covariance
        if cov[0] >= 0.0 and math.degrees(math.sqrt(max(cov[8], 0.0))) > self.max_imu_accuracy_deg:
            return None
        return quaternion_to_yaw(self.imu.orientation)

    def gps_course(self):
        # GPS course is noise when nearly stopped
        if self.gps_yaw is None or self.gps_speed is None or self.gps_speed < self.min_gps_speed_mps:
            return None
        return self.gps_yaw

    def init_state(self):
        # an exact origin means gps_to_xy hasn't seen a fix yet
        if self.gps_x is None or (self.gps_x == 0.0 and self.gps_y == 0.0):
            return False
        yaw = self.imu_yaw()
        if yaw is None:
            yaw = self.gps_course()
        if yaw is None:
            yaw = 0.0
        speed = self.gps_speed or 0.0
        self.state = np.array([self.gps_x, self.gps_y, speed * math.cos(yaw), speed * math.sin(yaw),
                               normalize_angle_rad(yaw), 0.0])
        self.cov = np.diag([1.0, 1.0, 1.0, 1.0, IMU_YAW_VAR, GYRO_BIAS_VAR])
        self.last_t = time.monotonic()
        return True

    def predict(self):
        now = time.monotonic()
        dt = max(1e-3, min(now - self.last_t, 0.2))
        self.last_t = now

        if self.imu is None:
            ax_body = ay_body = gyro_z = 0.0
        else:
            ax_body = self.imu.linear_acceleration.x
            ay_body = self.imu.linear_acceleration.y
            gyro_z = self.imu.angular_velocity.z

        x, y, vx, vy, yaw, gyro_bias = self.state
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        ax_world = ax_body * cos_yaw - ay_body * sin_yaw
        ay_world = ax_body * sin_yaw + ay_body * cos_yaw

        self.state[0] = x + vx * dt + 0.5 * ax_world * dt * dt
        self.state[1] = y + vy * dt + 0.5 * ay_world * dt * dt
        self.state[2] = vx + ax_world * dt
        self.state[3] = vy + ay_world * dt
        self.state[4] = normalize_angle_rad(yaw + (gyro_z - gyro_bias) * dt)

        # jacobian of the motion model
        dax_dyaw = -ax_body * sin_yaw - ay_body * cos_yaw
        day_dyaw = ax_body * cos_yaw - ay_body * sin_yaw
        f = np.eye(6)
        f[0, 2] = f[1, 3] = dt
        f[0, 4] = 0.5 * dax_dyaw * dt * dt
        f[1, 4] = 0.5 * day_dyaw * dt * dt
        f[2, 4] = dax_dyaw * dt
        f[3, 4] = day_dyaw * dt
        f[4, 5] = -dt
        q = np.diag([0.25 * ACCEL_VAR * dt ** 4, 0.25 * ACCEL_VAR * dt ** 4,
                     ACCEL_VAR * dt ** 2, ACCEL_VAR * dt ** 2,
                     GYRO_VAR * dt ** 2, GYRO_BIAS_VAR * dt])
        self.cov = f @ self.cov @ f.T + q

    def update(self, idx, z, var, is_angle=False):
        # measure state[idx] directly; standard kalman update
        h = np.zeros((len(idx), 6))
        h[range(len(idx)), idx] = 1.0
        residual = np.asarray(z, dtype=float) - h @ self.state
        if is_angle:
            residual[0] = normalize_angle_rad(residual[0])
        s = h @ self.cov @ h.T + np.eye(len(idx)) * var
        k = self.cov @ h.T @ np.linalg.inv(s)
        self.state = self.state + k @ residual
        self.state[4] = normalize_angle_rad(self.state[4])
        self.cov = (np.eye(6) - k @ h) @ self.cov

    def run(self):
        if self.state is None and not self.init_state():
            return

        self.predict()
        self.update([0, 1], [self.gps_x, self.gps_y], GPS_POSITION_VAR)
        course = self.gps_course()
        if course is not None:
            self.update([2, 3], [self.gps_speed * math.cos(course), self.gps_speed * math.sin(course)], GPS_VELOCITY_VAR)
        imu_yaw = self.imu_yaw()
        if imu_yaw is not None:
            self.update([4], [imu_yaw], IMU_YAW_VAR, is_angle=True)
        if course is not None:
            self.update([4], [course], GPS_YAW_VAR, is_angle=True)

        x, y, vx, vy, yaw, _ = self.state
        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = "map"
        odom.child_frame_id = "base_link"
        odom.pose.pose.position.x = float(x)
        odom.pose.pose.position.y = float(y)
        odom.pose.pose.orientation = yaw_to_quaternion(float(yaw))
        # twist is in the body frame
        odom.twist.twist.linear.x = float(vx * math.cos(yaw) + vy * math.sin(yaw))
        odom.twist.twist.linear.y = float(-vx * math.sin(yaw) + vy * math.cos(yaw))
        odom.pose.covariance[0] = float(self.cov[0, 0])
        odom.pose.covariance[7] = float(self.cov[1, 1])
        odom.pose.covariance[35] = float(self.cov[4, 4])
        odom.twist.covariance[0] = float(self.cov[2, 2])
        odom.twist.covariance[7] = float(self.cov[3, 3])
        self.pub.publish(odom)


def main(args=None):
    rclpy.init(args=args)
    node = HeadingFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
