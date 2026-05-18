import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
VENDORED_DONKEYCAR = REPO_ROOT / "static_donkeycar" / "donkeycar"
if str(VENDORED_DONKEYCAR) not in sys.path:
    sys.path.insert(0, str(VENDORED_DONKEYCAR))

import donkeycar as dk

from parts.gps_to_xy import GPS_to_xy
from parts.cte_controller import CTEController
from parts.heading_fusion import HeadingFusion
from parts.noop_part import NoOpPart
from sim.track_loader import load_track_data

import numpy as np

'''
checklist if not working

1. verify IMU, GPS, Main PCB UART are plugged in.
2. verify ports for each. should be /dev/tty* or /dev/usb*. unplug to test

run this to download the offline map of WL: 
```
python 
```
'''
import argparse


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("file_name", help="gps waypoint csv file name")
    parser.add_argument("--rate-hz", type=float, default=50.0)
    parser.add_argument("--max-loop-count", type=int, default=None)
    parser.add_argument("--no-telemetry", action="store_true")
    parser.add_argument("--no-logger", action="store_true")
    return parser


def parse_args(argv=None):
    return build_arg_parser().parse_args(argv)


def _instantiate_part(part_overrides, key, default_factory):
    override = (part_overrides or {}).get(key)
    if override is None:
        return default_factory()
    if callable(override):
        return override()
    return override


def _default_health_check():
    from parts.health_check import HealthCheck

    return HealthCheck("192.168.12.25", 6000)


def _default_uart():
    from parts.uart_backup import UART_backup_driver

    return UART_backup_driver("/dev/ttyACM2")


def _default_gps():
    from parts.gps import GPS

    return GPS('/dev/ttyACM1')


def _default_imu():
    from parts.bno086 import BNO086

    return BNO086(port='/dev/ttyACM0')


def _default_telemetry(enabled: bool):
    if not enabled:
        return NoOpPart()
    from parts.threaded_socket_pub_part import ThreadedTelemetryStreamer

    return ThreadedTelemetryStreamer()


def _default_logger(enabled: bool):
    if not enabled:
        return NoOpPart()
    from parts.logger_gps import Logger_GPS

    return Logger_GPS()


def build_vehicle(args, part_overrides=None, runtime_overrides=None):
    runtime_overrides = runtime_overrides or {}

    throttle = float(runtime_overrides.get("controller_throttle", 2500))
    kp, ki, kd = runtime_overrides.get("cte_pid", (0.3, 0.0, 0.3))
    y_kp, y_ki, y_kd = runtime_overrides.get("yaw_pid", (0.001, 0.0, 0.000001))

    track = load_track_data(args.file_name, default_throttle_pwm=throttle)
    if not track.has_geo or track.ref_lat_deg is None or track.ref_lon_deg is None:
        raise ValueError("cte_runner.py requires a geodetic waypoint CSV with latitude/longitude columns.")

    ref_lat0 = track.ref_lat_deg
    ref_lon0 = track.ref_lon_deg

    V = dk.vehicle.Vehicle()

    # Heart beat
    heartbeat = _instantiate_part(
        part_overrides,
        "health_check",
        _default_health_check,
    )
    V.add(heartbeat, inputs=[], outputs=["safety/heartbeat"])

    # # UART driver
    uart = _instantiate_part(
        part_overrides,
        "uart",
        _default_uart,
    )
    V.add(uart, inputs=["controls/throttle", "controls/steering", "safety/heartbeat"], outputs=[], threaded=False)

    gps = _instantiate_part(
        part_overrides,
        "gps",
        _default_gps,
    )
    V.add(gps, inputs=[], outputs=['lat_raw', 'lon_raw', 'alt', 'fix', 'corr_age', 'hdop', 'sat_count', 'gps_heading', 'gps_speed_mps'], threaded=True)

    # IMU
    imu = _instantiate_part(
        part_overrides,
        "imu",
        _default_imu,
    )
    V.add(imu, inputs=[], outputs=['imu_heading', 'imu_accuracy_deg', 'imu_lin_accel', 'imu_gyro'], threaded=True)

    # GPS to XY
    gps_to_xy = GPS_to_xy(ref_lat_deg=ref_lat0, ref_lon_deg=ref_lon0) # first point as origin
    V.add(gps_to_xy, inputs=["lat_raw", "lon_raw"], outputs=["x", "y", "gps_yaw"], threaded=False)

    heading_fusion = HeadingFusion()
    V.add(
        heading_fusion,
        inputs=["x", "y", "gps_yaw", "gps_speed_mps", "imu_heading", "imu_accuracy_deg", "imu_lin_accel", "imu_gyro"],
        outputs=["fused_x", "fused_y", "fused_yaw"],
        threaded=False,
    )

    # PID Controller
    # NOTE: must include "_throttle", with hardcoded throttle labels.
    csv_xy_path = str(track.cte_controller_csv)
    controller = CTEController(path_csv=csv_xy_path, throttle=throttle, kp=kp, ki=ki, kd=kd, kp_tangent=y_kp, ki_tangent=y_ki, kd_tangent=y_kd)

    V.add(controller, inputs=["fused_x", "fused_y", "fused_yaw"], outputs=["controls/throttle", "controls/steering"], threaded=False)

    telemetry_enabled = not args.no_telemetry and runtime_overrides.get("enable_telemetry", True)
    telemetry_part = _instantiate_part(
        part_overrides,
        "telemetry",
        lambda: _default_telemetry(telemetry_enabled),
    )
    V.add(telemetry_part, inputs=['lat_raw','lon_raw','fused_yaw', 'controls/steering'])

    logger_enabled = not args.no_logger and runtime_overrides.get("enable_logger", True)
    logger_part = _instantiate_part(
        part_overrides,
        "logger",
        lambda: _default_logger(logger_enabled),
    )
    V.add(logger_part, inputs=['lat_raw','lon_raw', 'controls/steering', 'controls/throttle', 'fix', 'gps_heading', 'gps_speed_mps', 'imu_heading', 'imu_accuracy_deg', 'fused_x', 'fused_y', 'fused_yaw'], outputs=[])

    context = {
        "track": track,
        "logger_part": logger_part,
        "rate_hz": float(runtime_overrides.get("rate_hz", args.rate_hz)),
        "max_loop_count": runtime_overrides.get("max_loop_count", args.max_loop_count),
        "controller_path": csv_xy_path,
    }
    return V, context


def main(argv=None):
    args = parse_args(argv)
    vehicle, context = build_vehicle(args)
    vehicle.start(rate_hz=context["rate_hz"], max_loop_count=context["max_loop_count"])


if __name__ == "__main__":
    main()
