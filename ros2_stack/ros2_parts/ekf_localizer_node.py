#!/usr/bin/env python3
# in: imu/data (Imu), gps/fix (NavSatFix) | out: gps/ekf_fix (NavSatFix), gps/ekf_vel (TwistStamped, ENU) from an EKF over [lat, lon, vx, vy]
import math
import time

import numpy as np
from filterpy.kalman import ExtendedKalmanFilter
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import Imu, NavSatFix, NavSatStatus

from ros2_parts.orientation import quaternion_to_yaw

R_EARTH = 6378137.0
RAD2DEG = 180.0 / np.pi
H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])  # we observe lat/lon directly


class EkfLocalizerNode(Node):
    def __init__(self):
        super().__init__("ekf_localizer")
        init_lat = self.declare_parameter("init_lat", 40.4237).value
        init_lon = self.declare_parameter("init_lon", -86.9212).value
        rate_hz = self.declare_parameter("rate_hz", 30.0).value

        self.ekf = ExtendedKalmanFilter(dim_x=4, dim_z=2)
        self.ekf.x = np.array([init_lat, init_lon, 0.0, 0.0])
        self.ekf.P = np.diag([1e-8, 1e-8, 1.0, 1.0])  # ~1e-5 deg ~= 1 m
        self.ekf.R = np.diag([4.5e-5 ** 2, 4.5e-5 ** 2])  # ~5 m GPS noise
        self.last_predict_time = None

        self.imu = None
        self.fix = None

        self.create_subscription(Imu, "imu/data", self.on_imu, qos_profile_sensor_data)
        self.create_subscription(NavSatFix, "gps/fix", self.on_fix, qos_profile_sensor_data)
        self.fix_pub = self.create_publisher(NavSatFix, "gps/ekf_fix", 10)
        self.vel_pub = self.create_publisher(TwistStamped, "gps/ekf_vel", 10)
        self.create_timer(1.0 / rate_hz, self.run)

    def on_imu(self, msg):
        self.imu = msg

    def on_fix(self, msg):
        if msg.status.status != NavSatStatus.STATUS_NO_FIX:
            self.fix = msg

    def predict(self, ax, ay, dt, accel_std=0.1):
        lat, lon, vx, vy = self.ekf.x
        cos_lat = max(np.cos(np.radians(lat)), 1e-6)
        sin_lat = np.sin(np.radians(lat))

        # jacobian of the motion model
        F = np.eye(4)
        F[0, 3] = dt / R_EARTH * RAD2DEG
        F[1, 0] = vx * dt * sin_lat / (R_EARTH * cos_lat ** 2)
        F[1, 2] = dt / (R_EARTH * cos_lat) * RAD2DEG

        # process noise, position part converted from m^2 to deg^2
        q_pos = (0.5 * accel_std * dt ** 2) ** 2
        q_vel = (accel_std * dt) ** 2
        Q = np.diag([q_pos / R_EARTH ** 2 * RAD2DEG ** 2,
                     q_pos / (R_EARTH * cos_lat) ** 2 * RAD2DEG ** 2, q_vel, q_vel])

        self.ekf.x = np.array([
            lat + (vy / R_EARTH) * dt * RAD2DEG,              # north velocity moves lat
            lon + (vx / (R_EARTH * cos_lat)) * dt * RAD2DEG,  # east velocity moves lon
            vx + ax * dt,
            vy + ay * dt,
        ])
        self.ekf.P = F @ self.ekf.P @ F.T + Q

    def run(self):
        now = time.monotonic()
        dt = 0.1 if self.last_predict_time is None else float(np.clip(now - self.last_predict_time, 1e-4, 0.2))
        self.last_predict_time = now

        # rotate body-frame accel into world east/north
        ax = ay = heading = 0.0
        if self.imu is not None:
            ax = self.imu.linear_acceleration.x
            ay = self.imu.linear_acceleration.y
            heading = quaternion_to_yaw(self.imu.orientation)
        ax_world = ax * math.cos(heading) - ay * math.sin(heading)
        ay_world = ax * math.sin(heading) + ay * math.cos(heading)

        self.predict(ax_world, ay_world, dt)
        if self.fix is not None:
            self.ekf.update(z=np.array([self.fix.latitude, self.fix.longitude]),
                            HJacobian=lambda x: H, Hx=lambda x: x[0:2])
            self.fix = None  # each fix corrects once

        lat, lon, vx, vy = self.ekf.x
        stamp = self.get_clock().now().to_msg()

        fix = NavSatFix()
        fix.header.stamp = stamp
        fix.header.frame_id = "gps"
        fix.status.status = NavSatStatus.STATUS_FIX
        fix.latitude = float(lat)
        fix.longitude = float(lon)
        self.fix_pub.publish(fix)

        vel = TwistStamped()
        vel.header.stamp = stamp
        vel.header.frame_id = "gps"
        vel.twist.linear.x = float(vx)
        vel.twist.linear.y = float(vy)
        self.vel_pub.publish(vel)


def main(args=None):
    rclpy.init(args=args)
    node = EkfLocalizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
