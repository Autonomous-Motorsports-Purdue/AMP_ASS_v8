from __future__ import annotations

import csv
import math
import threading
import time
from dataclasses import asdict
from pathlib import Path

from sim.kart_dynamics import CalibratedKartDynamics, KartDynamicsParams, TruthState
from sim.track_loader import TrackData, average_heading


class SharedTruthState:
    def __init__(self, track: TrackData, params: KartDynamicsParams, seed: int = 0, log_history: bool = True):
        self.track = track
        self.params = params
        self.dynamics = CalibratedKartDynamics(params=params, seed=seed)
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._log_history = bool(log_history)
        self._history: list[dict[str, float]] = []
        self._alive = True
        self._command_throttle = 0.0
        self._command_steering = 0.0

        heading_rad = average_heading(track.x_m, track.y_m)
        start_x = float(track.x_m[0] - params.start_offset_m * math.cos(heading_rad))
        start_y = float(track.y_m[0] - params.start_offset_m * math.sin(heading_rad))
        self.dynamics.reset(start_x, start_y, heading_rad, 0.0)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._update_loop, daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        with self._lock:
            self._alive = False
            self._command_throttle = 0.0
            self._command_steering = 0.0

    def set_command(self, throttle: float, steering: float, alive: bool = True) -> None:
        with self._lock:
            self._command_throttle = float(throttle)
            self._command_steering = float(steering)
            self._alive = bool(alive)

    def get_snapshot(self) -> TruthState:
        with self._lock:
            return self.dynamics.snapshot()

    def _update_loop(self) -> None:
        next_tick = time.monotonic()
        dt_s = 1.0 / max(self.params.sim_rate_hz, 1e-6)
        while self._running:
            now = time.monotonic()
            if now < next_tick:
                time.sleep(min(next_tick - now, 0.002))
                continue

            with self._lock:
                throttle = self._command_throttle if self._alive else 0.0
                steering = self._command_steering if self._alive else 0.0
                state = self.dynamics.step(dt_s=dt_s, throttle_cmd=throttle, steering_cmd=steering)
                if self._log_history:
                    self._history.append(asdict(state))
            next_tick += dt_s

    def dump_truth_log(self, output_csv: str | Path) -> Path:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if not self._history:
                writer.writerow(
                    [
                        "x_m",
                        "y_m",
                        "yaw_rad",
                        "speed_mps",
                        "yaw_rate_radps",
                        "steering_cmd",
                        "steering_effective",
                        "throttle_cmd",
                        "longitudinal_accel_mps2",
                        "lateral_accel_mps2",
                        "sim_time_s",
                    ]
                )
                return output_csv
            fieldnames = list(self._history[0].keys())
            writer.writerow(fieldnames)
            for row in self._history:
                writer.writerow([row[field] for field in fieldnames])
        return output_csv

