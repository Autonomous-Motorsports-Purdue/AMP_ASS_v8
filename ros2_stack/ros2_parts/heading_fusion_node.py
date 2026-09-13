#!/usr/bin/env python3
"""ROS 2 port of parts/heading_fusion.py.

Six-state EKF over [x, y, vx, vy, yaw, gyro_bias]. The prediction step
integrates the IMU; GPS position, GPS velocity and the two yaw sources are
folded in as separate measurement updates so each can be rejected on its own.
Every predict and update step is unchanged from the part.

Two units differ, because the ROS messages fix them: the gyro arrives in rad/s
rather than deg/s, and the IMU yaw arrives as an ENU orientation rather than a
compass heading. The filter math is untouched; only the conversions at the
boundary are gone.

subscribes: gps/pose (PoseStamped, local ENU position and GPS course),
            gps/vel (TwistStamped), imu/data (Imu)
publishes:  odometry/filtered (Odometry)
"""

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
from ros2_parts.parameters import declare

# yaw variance sits at index 8 of a row-major 3x3 orientation covariance
YAW_COVARIANCE_INDEX = 8


def is_valid_number(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


class HeadingFusionNode(Node):
    def __init__(self):
        super().__init__("heading_fusion")

        self.min_gps_speed_mps = declare(
            self, "min_gps_speed_mps", 1.0,
            "Speed below which the GPS course is too noisy to trust, in m/s.")
        self.max_imu_accuracy_deg = declare(
            self, "max_imu_accuracy_deg", 15.0,
            "Reported IMU accuracy above which its heading is ignored, in degrees.")
        self.gps_position_var = declare(
            self, "gps_position_std_m", 0.10, "GPS position noise, in meters.") ** 2
        self.gps_velocity_var = declare(
            self, "gps_velocity_std_mps", 0.10, "GPS velocity noise, in m/s.") ** 2
        self.gps_yaw_var = math.radians(declare(
            self, "gps_yaw_std_deg", 5.0, "GPS course noise, in degrees.")) ** 2
        self.imu_yaw_var = math.radians(declare(
            self, "imu_yaw_std_deg", 30.0, "IMU yaw noise, in degrees.")) ** 2
        self.accel_var = declare(
            self, "accel_std_mps2", 2.0, "Accelerometer noise, in m/s^2.") ** 2
        self.gyro_var = math.radians(declare(
            self, "gyro_std_dps", 1.5, "Gyro noise, in degrees per second.")) ** 2
        self.gyro_bias_var = math.radians(declare(
            self, "gyro_bias_std_dps", 0.1, "Gyro bias drift, in degrees per second.")) ** 2
        self.frame_id = declare(self, "frame_id", "map", "Frame the pose is expressed in.")
        self.child_frame_id = declare(
            self, "child_frame_id", "base_link", "Frame the velocity is expressed in.")
        rate_hz = declare(self, "rate_hz", 50.0, "Rate the filter steps at, in Hz.")

        self.state = None
        self.cov = None
        self.last_t = None

        # latest measurements, each replaced as it arrives
        self.gps_x = None
        self.gps_y = None
        self.gps_yaw_deg = None
        self.gps_speed_mps = None
        self.imu = None

        self.pub = self.create_publisher(Odometry, "odometry/filtered", 10)
        self.create_subscription(PoseStamped, "gps/pose", self.on_gps_pose, 10)
        self.create_subscription(
            TwistStamped, "gps/vel", self.on_gps_velocity, qos_profile_sensor_data)
        self.create_subscription(Imu, "imu/data", self.on_imu, qos_profile_sensor_data)
        self.create_timer(1.0 / rate_hz, self.run)

    def on_gps_pose(self, pose):
        self.gps_x = pose.pose.position.x
        self.gps_y = pose.pose.position.y
        self.gps_yaw_deg = math.degrees(quaternion_to_yaw(pose.pose.orientation))

    def on_gps_velocity(self, vel):
        self.gps_speed_mps = math.hypot(vel.twist.linear.x, vel.twist.linear.y)

    def on_imu(self, imu):
        self.imu = imu

    def _imu_yaw(self):
        if self.imu is None:
            return None

        cov = self.imu.orientation_covariance
        if cov[0] >= 0.0:
            accuracy_deg = math.degrees(math.sqrt(max(cov[YAW_COVARIANCE_INDEX], 0.0)))
            if accuracy_deg > self.max_imu_accuracy_deg:
                return None
        # a negative leading element means unknown accuracy; the part treated a
        # missing accuracy as usable, so it stays usable

        return math.degrees(quaternion_to_yaw(self.imu.orientation))

    def _gps_yaw(self):
        if not is_valid_number(self.gps_yaw_deg) or not is_valid_number(self.gps_speed_mps):
            return None
        if float(self.gps_speed_mps) < self.min_gps_speed_mps:
            return None
        return float(self.gps_yaw_deg)

    def _init_state(self):
        if not is_valid_number(self.gps_x) or not is_valid_number(self.gps_y):
            return False
        # an exact origin means the projection has not seen a fix yet
        if self.gps_x == 0.0 and self.gps_y == 0.0:
            return False

        yaw_deg = self._imu_yaw()
        if yaw_deg is None:
            yaw_deg = self._gps_yaw()
        if yaw_deg is None:
            yaw_deg = 0.0

        speed = float(self.gps_speed_mps) if is_valid_number(self.gps_speed_mps) else 0.0
        yaw_rad = math.radians(yaw_deg)
        self.state = np.array(
            [
                float(self.gps_x),
                float(self.gps_y),
                speed * math.cos(yaw_rad),
                speed * math.sin(yaw_rad),
                normalize_angle_rad(yaw_rad),
                0.0,
            ],
            dtype=float,
        )
        self.cov = np.diag([1.0, 1.0, 1.0, 1.0, self.imu_yaw_var, self.gyro_bias_var])
        self.last_t = time.monotonic()
        self.get_logger().info(
            f"initialized at ({self.gps_x:.2f}, {self.gps_y:.2f}), yaw {yaw_deg:.1f} deg")
        return True

    def _predict(self):
        now = time.monotonic()
        dt = 0.02 if self.last_t is None else now - self.last_t
        self.last_t = now
        dt = max(1e-3, min(float(dt), 0.2))

        if self.imu is None:
            ax_body = ay_body = gyro_z = 0.0
        else:
            ax_body = self.imu.linear_acceleration.x
            ay_body = self.imu.linear_acceleration.y
            # already rad/s, as sensor_msgs/Imu specifies
            gyro_z = self.imu.angular_velocity.z

        x, y, vx, vy, yaw, gyro_bias = self.state
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        ax_world = ax_body * cos_yaw - ay_body * sin_yaw
        ay_world = ax_body * sin_yaw + ay_body * cos_yaw
        omega = gyro_z - gyro_bias

        self.state[0] = x + vx * dt + 0.5 * ax_world * dt * dt
        self.state[1] = y + vy * dt + 0.5 * ay_world * dt * dt
        self.state[2] = vx + ax_world * dt
        self.state[3] = vy + ay_world * dt
        self.state[4] = normalize_angle_rad(yaw + omega * dt)

        dax_dyaw = -ax_body * sin_yaw - ay_body * cos_yaw
        day_dyaw = ax_body * cos_yaw - ay_body * sin_yaw
        f = np.eye(6)
        f[0, 2] = dt
        f[1, 3] = dt
        f[0, 4] = 0.5 * dax_dyaw * dt * dt
        f[1, 4] = 0.5 * day_dyaw * dt * dt
        f[2, 4] = dax_dyaw * dt
        f[3, 4] = day_dyaw * dt
        f[4, 5] = -dt

        q = np.diag(
            [
                0.25 * self.accel_var * dt ** 4,
                0.25 * self.accel_var * dt ** 4,
                self.accel_var * dt ** 2,
                self.accel_var * dt ** 2,
                self.gyro_var * dt ** 2,
                self.gyro_bias_var * dt,
            ]
        )
        self.cov = f @ self.cov @ f.T + q

    def _update(self, z, h, r, angle_index=None):
        z = np.asarray(z, dtype=float)
        h = np.asarray(h, dtype=float)
        r = np.asarray(r, dtype=float)
        residual = z - h @ self.state
        if angle_index is not None:
            residual[angle_index] = normalize_angle_rad(residual[angle_index])

        s = h @ self.cov @ h.T + r
        k = self.cov @ h.T @ np.linalg.inv(s)
        self.state = self.state + k @ residual
        self.state[4] = normalize_angle_rad(self.state[4])
        self.cov = (np.eye(6) - k @ h) @ self.cov

    def _update_position(self):
        if not is_valid_number(self.gps_x) or not is_valid_number(self.gps_y):
            return

        h = np.zeros((2, 6))
        h[0, 0] = 1.0
        h[1, 1] = 1.0
        r = np.diag([self.gps_position_var, self.gps_position_var])
        self._update([float(self.gps_x), float(self.gps_y)], h, r)

    def _update_gps_velocity(self):
        gps_yaw_deg = self._gps_yaw()
        if gps_yaw_deg is None:
            return

        speed = float(self.gps_speed_mps)
        yaw_rad = math.radians(gps_yaw_deg)
        h = np.zeros((2, 6))
        h[0, 2] = 1.0
        h[1, 3] = 1.0
        r = np.diag([self.gps_velocity_var, self.gps_velocity_var])
        self._update([speed * math.cos(yaw_rad), speed * math.sin(yaw_rad)], h, r)

    def _update_yaw(self, yaw_deg, yaw_var):
        if yaw_deg is None:
            return

        h = np.zeros((1, 6))
        h[0, 4] = 1.0
        self._update([math.radians(yaw_deg)], h, [[yaw_var]], angle_index=0)

    def run(self):
        if self.state is None and not self._init_state():
            return

        self._predict()
        self._update_position()
        self._update_gps_velocity()
        self._update_yaw(self._imu_yaw(), self.imu_yaw_var)
        self._update_yaw(self._gps_yaw(), self.gps_yaw_var)

        self.publish_odometry()

    def publish_odometry(self):
        yaw = float(self.state[4])

        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = self.frame_id
        odom.child_frame_id = self.child_frame_id
        odom.pose.pose.position.x = float(self.state[0])
        odom.pose.pose.position.y = float(self.state[1])
        odom.pose.pose.orientation = yaw_to_quaternion(yaw)

        # speed along the body's own forward axis, which is what Odometry
        # means by a twist in the child frame
        odom.twist.twist.linear.x = (
            float(self.state[2]) * math.cos(yaw) + float(self.state[3]) * math.sin(yaw))
        odom.twist.twist.linear.y = (
            -float(self.state[2]) * math.sin(yaw) + float(self.state[3]) * math.cos(yaw))

        # copy the filter's own covariance into the 6x6 pose block
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
