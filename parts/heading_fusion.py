import math
import time

import numpy as np


def is_valid_number(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def normalize_angle_deg(angle):
    return (float(angle) + 180.0) % 360.0 - 180.0


def normalize_angle_rad(angle):
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def compass_heading_to_math_yaw_deg(heading_deg):
    # BNO086 heading is compass-style: 0 north, 90 east, clockwise positive.
    # Controller yaw is math-style: 0 east, 90 north, counterclockwise positive.
    return normalize_angle_deg(90.0 - float(heading_deg))


def valid_tuple(values, size):
    if values is None:
        return None
    try:
        values = list(values)
    except TypeError:
        return None
    if len(values) < size:
        return None

    out = []
    for value in values[:size]:
        if not is_valid_number(value):
            return None
        out.append(float(value))
    return out


class HeadingFusion:
    def __init__(
        self,
        min_gps_speed_mps=1.0,
        max_imu_accuracy_deg=15.0,
        gps_position_std_m=0.10,
        gps_velocity_std_mps=0.10,
        gps_yaw_std_deg=5.0,
        imu_yaw_std_deg=30.0,
        accel_std_mps2=2.0,
        gyro_std_dps=1.5,
        gyro_bias_std_dps=0.1,
        verbose=False,
    ):
        self.min_gps_speed_mps = float(min_gps_speed_mps)
        self.max_imu_accuracy_deg = float(max_imu_accuracy_deg)
        self.gps_position_var = float(gps_position_std_m) ** 2
        self.gps_velocity_var = float(gps_velocity_std_mps) ** 2
        self.gps_yaw_var = math.radians(float(gps_yaw_std_deg)) ** 2
        self.imu_yaw_var = math.radians(float(imu_yaw_std_deg)) ** 2
        self.accel_var = float(accel_std_mps2) ** 2
        self.gyro_var = math.radians(float(gyro_std_dps)) ** 2
        self.gyro_bias_var = math.radians(float(gyro_bias_std_dps)) ** 2
        self.verbose = verbose

        self.state = None
        self.cov = None
        self.last_t = None

    def _imu_yaw(self, imu_heading, imu_accuracy_deg):
        if not is_valid_number(imu_heading):
            return None

        if (
            is_valid_number(imu_accuracy_deg)
            and float(imu_accuracy_deg) > self.max_imu_accuracy_deg
        ):
            return None

        return compass_heading_to_math_yaw_deg(imu_heading)

    def _gps_yaw(self, gps_yaw_deg, gps_speed_mps):
        if not is_valid_number(gps_yaw_deg) or not is_valid_number(gps_speed_mps):
            return None
        if float(gps_speed_mps) < self.min_gps_speed_mps:
            return None
        return normalize_angle_deg(gps_yaw_deg)

    def _init_state(self, gps_x, gps_y, gps_yaw_deg, gps_speed_mps, imu_heading, imu_accuracy_deg):
        if not is_valid_number(gps_x) or not is_valid_number(gps_y):
            return False
        if gps_x == 0.0 and gps_y == 0.0:
            return False

        imu_yaw_deg = self._imu_yaw(imu_heading, imu_accuracy_deg)
        yaw_deg = imu_yaw_deg
        if yaw_deg is None:
            yaw_deg = self._gps_yaw(gps_yaw_deg, gps_speed_mps)
        if yaw_deg is None:
            yaw_deg = 0.0

        speed = float(gps_speed_mps) if is_valid_number(gps_speed_mps) else 0.0
        yaw_rad = math.radians(yaw_deg)
        self.state = np.array(
            [
                float(gps_x),
                float(gps_y),
                speed * math.cos(yaw_rad),
                speed * math.sin(yaw_rad),
                normalize_angle_rad(yaw_rad),
                0.0,
            ],
            dtype=float,
        )
        self.cov = np.diag([1.0, 1.0, 1.0, 1.0, self.imu_yaw_var, self.gyro_bias_var])
        self.last_t = time.monotonic()
        return True

    def _predict(self, imu_lin_accel, imu_gyro):
        now = time.monotonic()
        dt = 0.02 if self.last_t is None else now - self.last_t
        self.last_t = now
        dt = max(1e-3, min(float(dt), 0.2))

        accel = valid_tuple(imu_lin_accel, 2)
        gyro = valid_tuple(imu_gyro, 3)
        ax_body = 0.0 if accel is None else accel[0]
        ay_body = 0.0 if accel is None else accel[1]
        gyro_z = 0.0 if gyro is None else math.radians(gyro[2])

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

    def _update_position(self, gps_x, gps_y):
        if not is_valid_number(gps_x) or not is_valid_number(gps_y):
            return

        h = np.zeros((2, 6))
        h[0, 0] = 1.0
        h[1, 1] = 1.0
        r = np.diag([self.gps_position_var, self.gps_position_var])
        self._update([float(gps_x), float(gps_y)], h, r)

    def _update_gps_velocity(self, gps_yaw_deg, gps_speed_mps):
        gps_yaw_deg = self._gps_yaw(gps_yaw_deg, gps_speed_mps)
        if gps_yaw_deg is None:
            return

        speed = float(gps_speed_mps)
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

    def run(
        self,
        gps_x,
        gps_y,
        gps_yaw_deg,
        gps_speed_mps,
        imu_heading,
        imu_accuracy_deg,
        imu_lin_accel,
        imu_gyro,
    ):
        if self.state is None and not self._init_state(
            gps_x,
            gps_y,
            gps_yaw_deg,
            gps_speed_mps,
            imu_heading,
            imu_accuracy_deg,
        ):
            return gps_x, gps_y, self._imu_yaw(imu_heading, imu_accuracy_deg)

        self._predict(imu_lin_accel, imu_gyro)
        self._update_position(gps_x, gps_y)
        self._update_gps_velocity(gps_yaw_deg, gps_speed_mps)
        self._update_yaw(self._imu_yaw(imu_heading, imu_accuracy_deg), self.imu_yaw_var)
        self._update_yaw(self._gps_yaw(gps_yaw_deg, gps_speed_mps), self.gps_yaw_var)

        if self.verbose:
            speed = math.hypot(self.state[2], self.state[3])
            print(
                "[HeadingFusion] gps_yaw:",
                None if not is_valid_number(gps_yaw_deg) else round(float(gps_yaw_deg), 2),
                "gps_speed:",
                None if not is_valid_number(gps_speed_mps) else round(float(gps_speed_mps), 2),
                "fused_xy:",
                (round(float(self.state[0]), 2), round(float(self.state[1]), 2)),
                "fused_speed:",
                round(float(speed), 2),
                "fused_yaw:",
                round(math.degrees(self.state[4]), 2),
            )

        return float(self.state[0]), float(self.state[1]), normalize_angle_deg(math.degrees(self.state[4]))

    def shutdown(self):
        pass