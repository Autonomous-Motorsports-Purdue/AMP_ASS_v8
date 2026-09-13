#!/usr/bin/env python3
"""ROS 2 port of parts/ekf_localizer.py.

Fuses GPS + IMU into [lat, lon, vx, vy] using an EKF. Expects IMU
acceleration in body frame plus a yaw heading, and rotates accel into world
East/North before prediction. Position is carried in degrees rather than
meters, so the process noise is converted from meters to degrees each step
using the current latitude.

The filter wants the heading in degrees. gps_mpc_runner.py passed it the IMU
part's yaw, which was in radians, so the body-to-world rotation was wrong.
It is now taken from the orientation quaternion in the right units.

subscribes: gps/fix (NavSatFix), imu/data (Imu)
publishes:  gps/ekf_fix (NavSatFix), gps/ekf_vel (TwistStamped, ENU)
"""

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
from ros2_parts.parameters import declare

R_EARTH = 6378137.0
RAD2DEG = 180.0 / np.pi


class EkfLocalizerNode(Node):
    def __init__(self):
        super().__init__("ekf_localizer")

        init_lat = declare(self, "init_lat", 40.4237, "Initial latitude estimate.")
        init_lon = declare(self, "init_lon", -86.9212, "Initial longitude estimate.")
        imu_rate = declare(self, "imu_rate", 10.0, "Nominal IMU rate, in Hz.")
        self.frame_id = declare(self, "frame_id", "gps", "Frame the antenna sits in.")
        gps_std_deg = declare(
            self, "gps_std_deg", 4.5e-5, "GPS position noise, in degrees (~5 m).")
        rate_hz = declare(self, "rate_hz", 30.0, "Rate the filter steps at, in Hz.")

        self.nominal_dt = 1.0 / imu_rate
        self.last_predict_time = None

        # EKF: state = [lat, lon, vx, vy]
        self.ekf = ExtendedKalmanFilter(dim_x=4, dim_z=2)
        self.ekf.x = np.array([init_lat, init_lon, 0.0, 0.0])

        # P: initial uncertainty
        # lat/lon in degrees (~1e-5 deg ~= 1m), velocity in m/s
        self.ekf.P = np.diag([1e-8, 1e-8, 1.0, 1.0])

        # R: GPS measurement noise
        self.ekf.R = np.diag([gps_std_deg**2, gps_std_deg**2])

        self.imu = None
        self.fix = None

        self.fix_pub = self.create_publisher(NavSatFix, "gps/ekf_fix", 10)
        self.vel_pub = self.create_publisher(TwistStamped, "gps/ekf_vel", 10)
        self.create_subscription(Imu, "imu/data", self.on_imu, qos_profile_sensor_data)
        self.create_subscription(NavSatFix, "gps/fix", self.on_fix, qos_profile_sensor_data)
        self.create_timer(1.0 / rate_hz, self.step)

    def on_imu(self, msg):
        self.imu = msg

    def on_fix(self, msg):
        if msg.status.status != NavSatStatus.STATUS_NO_FIX:
            self.fix = msg

    def _build_Q(self, dt, accel_std=0.1):
        """
        Process noise Q scales with dt.
        accel_std: expected IMU acceleration noise in m/s^2
        """
        q_pos = (0.5 * accel_std * dt**2) ** 2
        q_vel = (accel_std * dt) ** 2

        # Convert q_pos from m^2 to deg^2.
        cos_lat = np.cos(np.radians(self.ekf.x[0]))
        cos_lat = np.clip(cos_lat, 1e-6, None)
        q_lat = q_pos / R_EARTH**2 * (RAD2DEG**2)
        q_lon = q_pos / (R_EARTH * cos_lat) ** 2 * (RAD2DEG**2)

        return np.diag([q_lat, q_lon, q_vel, q_vel])

    def _f(self, x, dt, ax, ay):
        """State transition: propagate [lat, lon, vx, vy] forward by dt."""
        lat, lon, vx, vy = x
        cos_lat = np.cos(np.radians(lat))
        cos_lat = np.clip(cos_lat, 1e-6, None)
        return np.array([
            lat + (vy / R_EARTH) * dt * RAD2DEG,            # lat from vy (North)
            lon + (vx / (R_EARTH * cos_lat)) * dt * RAD2DEG,  # lon from vx (East)
            vx + ax * dt,                                   # vx from East accel
            vy + ay * dt,                                   # vy from North accel
        ])

    def _F_jacobian(self, x, dt):
        """Jacobian of _f with respect to state x."""
        lat, _, vx, _ = x
        cos_lat = np.cos(np.radians(lat))
        cos_lat = np.clip(cos_lat, 1e-6, None)
        sin_lat = np.sin(np.radians(lat))

        F = np.eye(4)
        F[0, 3] = dt / R_EARTH * RAD2DEG
        F[1, 0] = vx * dt * sin_lat / (R_EARTH * cos_lat**2)
        F[1, 2] = dt / (R_EARTH * cos_lat) * RAD2DEG
        return F

    def _h(self, x):
        """Measurement function: we observe [lat, lon] directly."""
        return x[0:2]

    def _H_jacobian(self, x):
        """Jacobian of _h constant since measurement model is linear."""
        H = np.zeros((2, 4))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        return H

    def _predict(self, ax, ay, dt):
        x = self.ekf.x
        F = self._F_jacobian(x, dt)
        Q = self._build_Q(dt)
        self.ekf.x = self._f(x, dt, ax, ay)
        self.ekf.P = F @ self.ekf.P @ F.T + Q

    def _update_gps(self, lat, lon):
        self.ekf.update(
            z=np.array([lat, lon]),
            HJacobian=self._H_jacobian,
            Hx=self._h,
        )

    def run(self, ax, ay, heading_rad, gps_lat, gps_lon):
        """
        ax, ay: body-frame acceleration in m/s^2
        heading_rad: yaw heading in radians
        """
        now = time.monotonic()
        if self.last_predict_time is None:
            dt = self.nominal_dt
        else:
            dt = now - self.last_predict_time
            dt = float(np.clip(dt, 1e-4, 0.2))
        self.last_predict_time = now

        cos_h = math.cos(heading_rad)
        sin_h = math.sin(heading_rad)
        ax_body = ax or 0.0
        ay_body = ay or 0.0
        ax_world = ax_body * cos_h - ay_body * sin_h
        ay_world = ax_body * sin_h + ay_body * cos_h

        self._predict(ax_world, ay_world, dt)
        if gps_lat is not None and gps_lon is not None:
            self._update_gps(gps_lat, gps_lon)
        return self.ekf.x

    def step(self):
        if self.imu is None:
            ax = ay = 0.0
            heading_rad = 0.0
        else:
            ax = self.imu.linear_acceleration.x
            ay = self.imu.linear_acceleration.y
            heading_rad = quaternion_to_yaw(self.imu.orientation)

        gps_lat = self.fix.latitude if self.fix is not None else None
        gps_lon = self.fix.longitude if self.fix is not None else None
        # each fix corrects once, the next step predicts only
        self.fix = None

        lat, lon, vx, vy = self.run(ax, ay, heading_rad, gps_lat, gps_lon)
        stamp = self.get_clock().now().to_msg()

        fix = NavSatFix()
        fix.header.stamp = stamp
        fix.header.frame_id = self.frame_id
        fix.status.status = NavSatStatus.STATUS_FIX
        fix.status.service = NavSatStatus.SERVICE_GPS
        fix.latitude = float(lat)
        fix.longitude = float(lon)
        fix.position_covariance[0] = float(self.ekf.P[0, 0])
        fix.position_covariance[4] = float(self.ekf.P[1, 1])
        fix.position_covariance_type = NavSatFix.COVARIANCE_TYPE_DIAGONAL_KNOWN
        self.fix_pub.publish(fix)

        vel = TwistStamped()
        vel.header.stamp = stamp
        vel.header.frame_id = self.frame_id
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
