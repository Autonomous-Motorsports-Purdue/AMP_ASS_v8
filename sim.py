#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import shlex
import sys
import time
import threading
from datetime import datetime
from pathlib import Path

from parts.noop_part import NoOpPart
from parts.sim_bno086 import SimBNO086
from parts.sim_gps import SimGPS
from parts.sim_health_check import SimHealthCheck
from parts.sim_uart import SimUARTBackupDriver
from sim.calibration import load_kart_params
from sim.shared_truth import SharedTruthState
from sim.track_loader import load_track_data
from sim.validation import compare_summaries, summarize_logger_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run a GPS/IMU kart runner against the simulator.')
    parser.add_argument("runner_command", help='quoted runner command, e.g. "cte_runner.py gps_paths/path.csv"')
    parser.add_argument("--params", default=None, help="optional sim/kart_params.json override")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--controller-throttle", type=float, default=2500.0, help="constant throttle PWM to write into the generated cte controller path")
    parser.add_argument("--max-time-s", type=float, default=None)
    parser.add_argument("--stop-at-lap", action="store_true", help="stop automatically when the simulated kart completes one lap")
    parser.add_argument("--log-dir", default="sim_outputs")
    parser.add_argument("--reference-log", default=None, help="optional real logger CSV for summary comparison")
    return parser.parse_args()


def normalize_runner_tokens(command: str) -> list[str]:
    tokens = shlex.split(command)
    if not tokens:
        raise ValueError("Runner command is empty.")
    if tokens[0] in {"python", "python3"}:
        tokens = tokens[1:]
    if not tokens:
        raise ValueError("Runner command did not contain a runner script.")
    return tokens


def load_runner_module(script_name: str):
    stem = Path(script_name).name
    if stem == "cte_runner.py":
        import cte_runner as runner_module

        return runner_module
    raise ValueError(f"Unsupported simulated runner: {script_name}. Only cte_runner.py is supported right now.")


def estimate_timeout_s(track, params, controller_throttle: float) -> float:
    expected_speed = max(
        0.5,
        params.speed_intercept_mps + params.speed_slope_mps_per_erpm * float(controller_throttle),
    )
    return max(20.0, 2.5 * float(track.path_length_m) / expected_speed)


def build_lap_monitor(vehicle, shared_truth, track):
    def _run():
        start_position = None
        last_position = None
        distance_traveled_m = 0.0
        completion_radius_m = 5.0
        min_distance_m = 0.75 * float(track.path_length_m)
        while vehicle.on:
            snapshot = shared_truth.get_snapshot()
            current_position = (float(snapshot.x_m), float(snapshot.y_m))
            if start_position is None:
                start_position = current_position
            if last_position is not None:
                distance_traveled_m += math.hypot(
                    current_position[0] - last_position[0],
                    current_position[1] - last_position[1],
                )
            last_position = current_position
            if (
                distance_traveled_m >= min_distance_m
                and start_position is not None
                and math.hypot(
                    current_position[0] - start_position[0],
                    current_position[1] - start_position[1],
                ) <= completion_radius_m
            ):
                vehicle.on = False
                return
            time.sleep(0.05)

    return threading.Thread(target=_run, daemon=True)


def main() -> None:
    args = parse_args()
    runner_tokens = normalize_runner_tokens(args.runner_command)
    runner_module = load_runner_module(runner_tokens[0])
    runner_args = runner_module.parse_args(runner_tokens[1:])

    params = load_kart_params(args.params)
    controller_throttle = float(args.controller_throttle)
    track = load_track_data(runner_args.file_name, default_throttle_pwm=controller_throttle)

    log_root = Path(args.log_dir).expanduser().resolve()
    log_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    truth_log_path = log_root / f"{stamp}_truth.csv"
    summary_path = log_root / f"{stamp}_summary.json"

    shared_truth = SharedTruthState(track=track, params=params, seed=args.seed, log_history=True)

    part_overrides = {
        "health_check": lambda: SimHealthCheck(),
        "uart": lambda: SimUARTBackupDriver(shared_truth=shared_truth, params=params),
        "gps": lambda: SimGPS(shared_truth=shared_truth, params=params, gps_frame=track.gps_frame, seed=args.seed + 1),
        "imu": lambda: SimBNO086(shared_truth=shared_truth, params=params, seed=args.seed + 2),
        "telemetry": lambda: NoOpPart(),
    }
    runtime_overrides = {
        "controller_throttle": controller_throttle,
        "enable_telemetry": False,
        "rate_hz": params.drive_loop_hz,
    }

    vehicle, context = runner_module.build_vehicle(runner_args, part_overrides=part_overrides, runtime_overrides=runtime_overrides)

    max_time_s = args.max_time_s if args.max_time_s is not None else estimate_timeout_s(track, params, controller_throttle)
    max_loop_count = int(math.ceil(max_time_s * float(context["rate_hz"])))

    shared_truth.start()
    lap_monitor = None
    if args.stop_at_lap and track.is_loop:
        lap_monitor = build_lap_monitor(vehicle, shared_truth, track)
        lap_monitor.start()
    time.sleep(0.05)
    try:
        loop_count, loop_time = vehicle.start(rate_hz=context["rate_hz"], max_loop_count=max_loop_count)
    finally:
        shared_truth.shutdown()

    shared_truth.dump_truth_log(truth_log_path)

    logger_part = context.get("logger_part")
    runner_log_path = None
    if logger_part is not None and hasattr(logger_part, "info_csv"):
        runner_log_path = str(Path(logger_part.info_csv).resolve())

    payload = {
        "runner_command": args.runner_command,
        "controller_throttle": controller_throttle,
        "runner_log_path": runner_log_path,
        "truth_log_path": str(truth_log_path),
        "loop_count": loop_count,
        "loop_time_s": loop_time,
        "max_time_s": max_time_s,
        "params_path": args.params,
    }

    if runner_log_path is not None:
        payload["sim_summary"] = summarize_logger_csv(runner_log_path, runner_args.file_name)
    if args.reference_log and runner_log_path is not None:
        real_summary = summarize_logger_csv(args.reference_log, runner_args.file_name)
        sim_summary = payload["sim_summary"]
        payload["reference_summary"] = real_summary
        payload["comparison"] = compare_summaries(real_summary, sim_summary)

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

