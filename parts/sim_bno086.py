from __future__ import annotations

import math
import threading
import time

import numpy as np

from sim.kart_dynamics import KartDynamicsParams, math_yaw_to_compass_heading
from sim.shared_truth import SharedTruthState


class SimBNO086:
    def __init__(
        self,
        shared_truth: SharedTruthState,
        params: KartDynamicsParams,
        seed: int = 0,
        donkey: bool = True,
        *args,
        **kwargs,
    ):
        self.shared_truth = shared_truth
        self.params = params
        self.donkey = donkey
        self.rng = np.random.default_rng(seed)
        self.running = True
        self.lock = threading.Lock()
        self.latest = None

    def _sample_once(self):
        state = self.shared_truth.get_snapshot()
        heading_deg = math_yaw_to_compass_heading(
            math.degrees(state.yaw_rad + self.rng.normal(0.0, math.radians(self.params.imu_heading_sigma_deg)))
        )
        accuracy_deg = max(
            0.5,
            float(self.params.imu_accuracy_deg_mean + self.rng.normal(0.0, self.params.imu_accuracy_deg_std)),
        )
        lin_ax = float(state.longitudinal_accel_mps2 + self.rng.normal(0.0, self.params.imu_accel_sigma_mps2))
        lin_ay = float(state.lateral_accel_mps2 + self.rng.normal(0.0, self.params.imu_accel_sigma_mps2))
        gyro_z_dps = float(math.degrees(state.yaw_rate_radps) + self.rng.normal(0.0, self.params.imu_gyro_sigma_dps))

        if self.donkey:
            sample = [
                float(heading_deg),
                float(accuracy_deg),
                (lin_ax, lin_ay, 0.0),
                (0.0, 0.0, gyro_z_dps),
            ]
        else:
            sample = {
                "heading": float(heading_deg),
                "accuracy_deg": float(accuracy_deg),
                "lin_accel": (lin_ax, lin_ay, 0.0),
                "gyro_dps": (0.0, 0.0, gyro_z_dps),
            }
        with self.lock:
            self.latest = sample

    def update(self):
        period_s = 1.0 / max(self.params.imu_rate_hz, 1e-6)
        next_tick = time.monotonic()
        while self.running:
            now = time.monotonic()
            if now >= next_tick:
                self._sample_once()
                next_tick += period_s
            time.sleep(min(max(0.0, next_tick - time.monotonic()), 0.001))

    def run(self):
        with self.lock:
            return self.latest

    def run_threaded(self):
        return self.run()

    def shutdown(self):
        self.running = False

