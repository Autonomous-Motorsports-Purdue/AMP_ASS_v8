from __future__ import annotations

import math
import threading
import time

import numpy as np

from parts.gps_to_xy import GPS_to_xy
from sim.kart_dynamics import KartDynamicsParams, math_yaw_to_compass_heading
from sim.shared_truth import SharedTruthState


class SimGPS:
    def __init__(
        self,
        shared_truth: SharedTruthState,
        params: KartDynamicsParams,
        gps_frame: GPS_to_xy | None = None,
        seed: int = 0,
        return_dict: bool = False,
        debug: bool = False,
    ):
        self.shared_truth = shared_truth
        self.params = params
        self.gps_frame = gps_frame or shared_truth.track.gps_frame or GPS_to_xy(40.0, -86.0)
        self.return_dict = return_dict
        self.debug = debug
        self.rng = np.random.default_rng(seed)
        self.running = True
        self.lock = threading.Lock()
        self._bias_e_m = 0.0
        self._bias_n_m = 0.0
        self.latest_output = {
            "lat": None,
            "lon": None,
            "alt": params.altitude_m,
            "fix": "NO FIX",
            "diff_age": None,
            "hdop": None,
            "num_sv": None,
            "course_deg": None,
            "speed_mps": None,
        }

    def _sample_fix(self) -> str:
        labels = list(self.params.gps_fix_probabilities.keys())
        probs = np.asarray([self.params.gps_fix_probabilities[label] for label in labels], dtype=float)
        probs = probs / probs.sum()
        return str(self.rng.choice(labels, p=probs))

    def _update_once(self):
        state = self.shared_truth.get_snapshot()
        fix = self._sample_fix()
        sigma = float(self.params.gps_position_sigma_m.get(fix, 1.0))
        white_frac = float(np.clip(self.params.gps_white_noise_fraction, 0.0, 1.0))
        tau_s = max(float(self.params.gps_noise_correlation_s), 1e-3)
        dt_s = 1.0 / max(self.params.gps_rate_hz, 1e-6)
        alpha = float(np.exp(-dt_s / tau_s))
        sigma_bias = sigma * math.sqrt(max(0.0, 1.0 - white_frac * white_frac))
        sigma_white = sigma * white_frac
        bias_drive_std = sigma_bias * math.sqrt(max(0.0, 1.0 - alpha * alpha))
        self._bias_e_m = alpha * self._bias_e_m + float(self.rng.normal(0.0, bias_drive_std))
        self._bias_n_m = alpha * self._bias_n_m + float(self.rng.normal(0.0, bias_drive_std))
        noise_e = self._bias_e_m + float(self.rng.normal(0.0, sigma_white))
        noise_n = self._bias_n_m + float(self.rng.normal(0.0, sigma_white))
        lat, lon = self.gps_frame.to_latlon(state.x_m + noise_e, state.y_m + noise_n)

        speed = max(0.0, state.speed_mps + float(self.rng.normal(0.0, self.params.gps_speed_sigma_mps)))
        course_math_deg = math.degrees(state.yaw_rad + float(self.rng.normal(0.0, math.radians(self.params.gps_heading_sigma_deg))))
        course_deg = math_yaw_to_compass_heading(course_math_deg)

        if fix == "RTK FIXED":
            hdop = 0.5
            num_sv = 22
            diff_age = 0.1
        elif fix == "RTK FLOAT":
            hdop = 0.9
            num_sv = 18
            diff_age = 0.8
        elif fix == "3D":
            hdop = 1.6
            num_sv = 12
            diff_age = None
        else:
            hdop = 99.0
            num_sv = 0
            diff_age = None

        with self.lock:
            self.latest_output = {
                "lat": float(lat),
                "lon": float(lon),
                "alt": float(self.params.altitude_m),
                "fix": fix,
                "diff_age": diff_age,
                "hdop": hdop,
                "num_sv": num_sv,
                "course_deg": float(course_deg),
                "speed_mps": float(speed),
            }

    def update(self):
        period_s = 1.0 / max(self.params.gps_rate_hz, 1e-6)
        next_tick = time.monotonic()
        while self.running:
            now = time.monotonic()
            if now >= next_tick:
                self._update_once()
                next_tick += period_s
            time.sleep(min(max(0.0, next_tick - time.monotonic()), 0.002))

    def _to_legacy_tuple(self, out):
        return (
            out.get("lat"),
            out.get("lon"),
            out.get("alt"),
            out.get("fix"),
            out.get("diff_age"),
            out.get("hdop"),
            out.get("num_sv"),
            out.get("course_deg"),
            out.get("speed_mps"),
        )

    def run(self):
        with self.lock:
            out = dict(self.latest_output)
        return out if self.return_dict else self._to_legacy_tuple(out)

    def run_threaded(self):
        return self.run()

    def shutdown(self):
        self.running = False

