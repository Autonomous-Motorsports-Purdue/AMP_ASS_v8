"""Run one open-loop system-ID test case from a YAML plan.

    python sysid_runner.py plans/day1.yaml steer_step [--dry-run]

Abort with Ctrl-C: the UART driver's shutdown sends throttle 0, steer 0.
"""
import argparse
import datetime
import shutil

import donkeycar as dk
import yaml

from parts.bno086 import BNO086
from parts.channel_logger import ChannelLogger
from parts.gps import GPS
from parts.health_check import HealthCheck
from parts.loop_clock import LoopClock
from parts.sysid_sequencer import SysIDSequencer
from parts.uart_backup import UART_backup_driver


class PrintDriver:
    def run(self, v, s, alive, fix):
        print(f"T:{v} S:{s:+.2f} fix:{fix}")
        return s, v


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("plan")
    parser.add_argument("test")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    plan = yaml.safe_load(open(args.plan))
    rate_hz = plan.get("rate_hz", 50)
    segments = [{"t": 3}] + plan["tests"][args.test]  # 3 s stationary baseline
    out = f"data/sysid/{datetime.datetime.now():%Y-%m-%d-%H-%M-%S}_{args.test}"

    V = dk.vehicle.Vehicle()
    V.add(LoopClock(), outputs=["loop/index", "loop/monotonic_ns", "loop/wall_time"])
    V.add(HealthCheck("192.168.12.25", 6000), outputs=["safety/heartbeat"])
    V.add(GPS("/dev/ttyACM0"), outputs=["lat", "lon", "alt", "fix", "corr_age", "hdop", "sat_count", "gps_heading", "gps_speed"], threaded=True)
    V.add(BNO086(port="/dev/ttyACM2", raw_log_path=out + "_imu.csv"),
          outputs=["imu_heading", "imu_accuracy_deg", "imu_lin_accel", "imu_gyro", "imu_seq", "imu_t_ms", "imu_rx_ns"], threaded=True)

    sequencer = SysIDSequencer(segments, plan.get("steer_bias", 0.0), plan.get("max_speed_mps", 6.0))
    V.add(sequencer, inputs=["loop/monotonic_ns", "gps_speed"],
          outputs=["controls/throttle", "controls/steering", "sysid/segment", "sysid/t_in_segment"])

    driver = PrintDriver() if args.dry_run else UART_backup_driver("/dev/ttyACM1")
    V.add(driver, inputs=["controls/throttle", "controls/steering", "safety/heartbeat", "fix"],
          outputs=["commanded/steer", "commanded/throttle"])

    logger = ChannelLogger([
        "loop/index", "loop/monotonic_ns", "lat", "lon", "fix", "gps_heading", "gps_speed",
        "imu_heading", "imu_accuracy_deg", ("imu_lin_accel", 3), ("imu_gyro", 3), "imu_seq", "imu_t_ms", "imu_rx_ns",
        "controls/throttle", "controls/steering", "commanded/steer", "commanded/throttle",
        "sysid/segment", "sysid/t_in_segment",
    ], out + ".csv")
    V.add(logger, inputs=logger.inputs)
    shutil.copy(args.plan, out + ".yaml")

    input(f"{args.test}: {sequencer.duration_s:.0f} s. Confirm RTK fix, then press Enter to arm. ")
    V.start(rate_hz=rate_hz, max_loop_count=int(sequencer.duration_s * rate_hz))
