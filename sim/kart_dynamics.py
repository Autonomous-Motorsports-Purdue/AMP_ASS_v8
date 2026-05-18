from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def normalize_angle_rad(angle_rad: float) -> float:
    return (float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi


def math_yaw_to_compass_heading(math_yaw_deg: float) -> float:
    heading = 90.0 - float(math_yaw_deg)
    heading %= 360.0
    if heading < 0.0:
        heading += 360.0
    return heading


@dataclass
class KartDynamicsParams:
    drive_loop_hz: float = 50.0
    sim_rate_hz: float = 200.0
    gps_rate_hz: float = 10.0
    imu_rate_hz: float = 50.0
    start_offset_m: float = 0.35
    speed_intercept_mps: float = 0.05
    speed_slope_mps_per_erpm: float = 0.000803
    speed_lag_s: float = 0.60
    max_speed_mps: float = 5.0
    steering_trim: float = 0.097
    curvature_intercept_inv_m: float = 0.0217
    curvature_gain_inv_m: float = -0.223
    curvature_cubic_inv_m: float = 0.0
    steering_delay_s: float = 0.46
    steering_lag_s: float = 0.15
    steering_rate_limit_per_s: float = 4.0
    gps_fix_probabilities: dict[str, float] = field(
        default_factory=lambda: {"RTK FIXED": 0.90, "RTK FLOAT": 0.03, "3D": 0.06, "NO FIX": 0.01}
    )
    gps_position_sigma_m: dict[str, float] = field(
        default_factory=lambda: {"RTK FIXED": 0.02, "RTK FLOAT": 0.35, "3D": 1.00, "NO FIX": 3.00}
    )
    gps_noise_correlation_s: float = 1.5
    gps_white_noise_fraction: float = 0.15
    gps_speed_sigma_mps: float = 0.08
    gps_heading_sigma_deg: float = 3.0
    imu_heading_sigma_deg: float = 1.5
    imu_accuracy_deg_mean: float = 5.5
    imu_accuracy_deg_std: float = 0.7
    imu_accel_sigma_mps2: float = 0.10
    imu_gyro_sigma_dps: float = 0.35
    altitude_m: float = 0.0
    startup_warmup_loops: int = 10
    startup_clip_duration_s: float = 4.0
    startup_clip_throttle: float = 1500.0

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "KartDynamicsParams":
        defaults = cls()
        return cls(
            drive_loop_hz=float(payload.get("drive_loop_hz", defaults.drive_loop_hz)),
            sim_rate_hz=float(payload.get("sim_rate_hz", defaults.sim_rate_hz)),
            gps_rate_hz=float(payload.get("gps_rate_hz", defaults.gps_rate_hz)),
            imu_rate_hz=float(payload.get("imu_rate_hz", defaults.imu_rate_hz)),
            start_offset_m=float(payload.get("start_offset_m", defaults.start_offset_m)),
            speed_intercept_mps=float(payload.get("speed_intercept_mps", defaults.speed_intercept_mps)),
            speed_slope_mps_per_erpm=float(payload.get("speed_slope_mps_per_erpm", defaults.speed_slope_mps_per_erpm)),
            speed_lag_s=float(payload.get("speed_lag_s", defaults.speed_lag_s)),
            max_speed_mps=float(payload.get("max_speed_mps", defaults.max_speed_mps)),
            steering_trim=float(payload.get("steering_trim", defaults.steering_trim)),
            curvature_intercept_inv_m=float(payload.get("curvature_intercept_inv_m", defaults.curvature_intercept_inv_m)),
            curvature_gain_inv_m=float(payload.get("curvature_gain_inv_m", defaults.curvature_gain_inv_m)),
            curvature_cubic_inv_m=float(payload.get("curvature_cubic_inv_m", defaults.curvature_cubic_inv_m)),
            steering_delay_s=float(payload.get("steering_delay_s", defaults.steering_delay_s)),
            steering_lag_s=float(payload.get("steering_lag_s", defaults.steering_lag_s)),
            steering_rate_limit_per_s=float(payload.get("steering_rate_limit_per_s", defaults.steering_rate_limit_per_s)),
            gps_fix_probabilities=dict(payload.get("gps_fix_probabilities", defaults.gps_fix_probabilities)),
            gps_position_sigma_m=dict(payload.get("gps_position_sigma_m", defaults.gps_position_sigma_m)),
            gps_noise_correlation_s=float(payload.get("gps_noise_correlation_s", defaults.gps_noise_correlation_s)),
            gps_white_noise_fraction=float(payload.get("gps_white_noise_fraction", defaults.gps_white_noise_fraction)),
            gps_speed_sigma_mps=float(payload.get("gps_speed_sigma_mps", defaults.gps_speed_sigma_mps)),
            gps_heading_sigma_deg=float(payload.get("gps_heading_sigma_deg", defaults.gps_heading_sigma_deg)),
            imu_heading_sigma_deg=float(payload.get("imu_heading_sigma_deg", defaults.imu_heading_sigma_deg)),
            imu_accuracy_deg_mean=float(payload.get("imu_accuracy_deg_mean", defaults.imu_accuracy_deg_mean)),
            imu_accuracy_deg_std=float(payload.get("imu_accuracy_deg_std", defaults.imu_accuracy_deg_std)),
            imu_accel_sigma_mps2=float(payload.get("imu_accel_sigma_mps2", defaults.imu_accel_sigma_mps2)),
            imu_gyro_sigma_dps=float(payload.get("imu_gyro_sigma_dps", defaults.imu_gyro_sigma_dps)),
            altitude_m=float(payload.get("altitude_m", defaults.altitude_m)),
            startup_warmup_loops=int(payload.get("startup_warmup_loops", defaults.startup_warmup_loops)),
            startup_clip_duration_s=float(payload.get("startup_clip_duration_s", defaults.startup_clip_duration_s)),
            startup_clip_throttle=float(payload.get("startup_clip_throttle", defaults.startup_clip_throttle)),
        )


@dataclass
class TruthState:
    x_m: float
    y_m: float
    yaw_rad: float
    speed_mps: float
    yaw_rate_radps: float
    steering_cmd: float
    steering_effective: float
    throttle_cmd: float
    longitudinal_accel_mps2: float
    lateral_accel_mps2: float
    sim_time_s: float


class CalibratedKartDynamics:
    def __init__(self, params: KartDynamicsParams, seed: int = 0):
        self.params = params
        self.rng = np.random.default_rng(seed)
        self._steering_buffer: list[float] = []
        self.reset(0.0, 0.0, 0.0, 0.0)

    def reset(self, x_m: float, y_m: float, yaw_rad: float, speed_mps: float = 0.0) -> None:
        self.x_m = float(x_m)
        self.y_m = float(y_m)
        self.yaw_rad = float(yaw_rad)
        self.speed_mps = float(speed_mps)
        self.yaw_rate_radps = 0.0
        self.steering_cmd = 0.0
        self.steering_effective = 0.0
        self.throttle_cmd = 0.0
        self.longitudinal_accel_mps2 = 0.0
        self.lateral_accel_mps2 = 0.0
        self.sim_time_s = 0.0
        delay_steps = max(1, int(round(self.params.steering_delay_s * self.params.sim_rate_hz)))
        self._steering_buffer = [0.0] * delay_steps

    def _apply_steering_dynamics(self, steering_cmd: float, dt_s: float) -> float:
        limited_target = float(np.clip(steering_cmd, -1.0, 1.0))
        max_delta = float(self.params.steering_rate_limit_per_s) * dt_s
        delta = float(np.clip(limited_target - self.steering_cmd, -max_delta, max_delta))
        self.steering_cmd += delta

        self._steering_buffer.append(self.steering_cmd)
        delayed_cmd = self._steering_buffer.pop(0)

        alpha = min(1.0, dt_s / max(self.params.steering_lag_s, dt_s))
        self.steering_effective += (delayed_cmd - self.steering_effective) * alpha
        self.steering_effective = float(np.clip(self.steering_effective, -1.0, 1.0))
        return self.steering_effective

    def _target_speed(self, throttle_cmd: float) -> float:
        target = self.params.speed_intercept_mps + self.params.speed_slope_mps_per_erpm * float(throttle_cmd)
        return float(np.clip(target, 0.0, self.params.max_speed_mps))

    def step(self, dt_s: float, throttle_cmd: float, steering_cmd: float) -> TruthState:
        dt_s = max(1e-4, float(dt_s))
        effective_steering = self._apply_steering_dynamics(steering_cmd, dt_s)
        self.throttle_cmd = float(throttle_cmd)

        target_speed = self._target_speed(throttle_cmd)
        speed_alpha = min(1.0, dt_s / max(self.params.speed_lag_s, dt_s))
        prev_speed = self.speed_mps
        self.speed_mps += (target_speed - self.speed_mps) * speed_alpha
        self.speed_mps = max(0.0, min(self.params.max_speed_mps, self.speed_mps))
        self.longitudinal_accel_mps2 = (self.speed_mps - prev_speed) / dt_s

        curvature = (
            self.params.curvature_intercept_inv_m
            + self.params.curvature_gain_inv_m * effective_steering
            + self.params.curvature_cubic_inv_m * (effective_steering ** 3)
        )
        self.yaw_rate_radps = self.speed_mps * curvature
        self.lateral_accel_mps2 = self.speed_mps * self.yaw_rate_radps

        self.yaw_rad = normalize_angle_rad(self.yaw_rad + self.yaw_rate_radps * dt_s)
        self.x_m += self.speed_mps * math.cos(self.yaw_rad) * dt_s
        self.y_m += self.speed_mps * math.sin(self.yaw_rad) * dt_s
        self.sim_time_s += dt_s

        return self.snapshot()

    def snapshot(self) -> TruthState:
        return TruthState(
            x_m=float(self.x_m),
            y_m=float(self.y_m),
            yaw_rad=float(self.yaw_rad),
            speed_mps=float(self.speed_mps),
            yaw_rate_radps=float(self.yaw_rate_radps),
            steering_cmd=float(self.steering_cmd),
            steering_effective=float(self.steering_effective),
            throttle_cmd=float(self.throttle_cmd),
            longitudinal_accel_mps2=float(self.longitudinal_accel_mps2),
            lateral_accel_mps2=float(self.lateral_accel_mps2),
            sim_time_s=float(self.sim_time_s),
        )

