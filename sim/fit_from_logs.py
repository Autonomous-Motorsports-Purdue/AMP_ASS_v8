from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim.calibration import save_kart_params


DATA_RE = re.compile(r"data0*(\d+)\.csv$", re.IGNORECASE)
DEFAULT_OUTPUT = Path(__file__).with_name("kart_params.json")


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def parse_timestamp(value: str) -> float:
    return datetime.strptime(value, "%Y-%m-%d-%H-%M-%S.%f").timestamp()


def allowed_logs(aks_dir: Path) -> list[Path]:
    outputs = []
    for path in sorted(aks_dir.glob("data*.csv")):
        match = DATA_RE.match(path.name)
        if not match:
            continue
        if int(match.group(1)) >= 5:
            outputs.append(path)
    return outputs


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                row["_t"] = parse_timestamp(row["timestamp"])
            except Exception:
                continue
            rows.append(row)
    return rows


def numeric_array(rows: list[dict], key: str, default=np.nan) -> np.ndarray:
    values = []
    for row in rows:
        raw = row.get(key, "")
        if raw in ("", None):
            values.append(default)
        else:
            try:
                values.append(float(raw))
            except Exception:
                values.append(default)
    return np.asarray(values, dtype=float)


def text_array(rows: list[dict], key: str) -> np.ndarray:
    return np.asarray([row.get(key, "") for row in rows], dtype=object)


def weighted_linear_fit(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    x_mat = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.lstsq(w[:, None] * x_mat, w * y, rcond=None)[0]
    pred = x_mat @ beta
    y_bar = np.average(y, weights=w)
    sst = np.sum(w * (y - y_bar) ** 2)
    ssr = np.sum(w * (y - pred) ** 2)
    r2 = 1.0 - ssr / sst if sst > 0 else 0.0
    return beta, float(r2)


def _gps_speed_plateaus(rows: list[dict]) -> list[tuple[int, int, float, float]]:
    t = np.asarray([row["_t"] for row in rows], dtype=float)
    steering = np.abs(numeric_array(rows, "steering"))
    throttle = numeric_array(rows, "throttle")
    gps_speed = numeric_array(rows, "gps_speed")
    fix = text_array(rows, "fix")
    throttle_s = moving_average(throttle, 11)
    throttle_rate = np.gradient(throttle_s, t)
    valid = (
        np.isfinite(gps_speed)
        & np.isin(fix, ["RTK FIXED", "3D"])
        & (steering < 0.15)
        & (np.abs(throttle_rate) < 250.0)
    )
    results = []
    for target in sorted({int(round(x / 50.0) * 50) for x in throttle[np.isfinite(throttle)]}):
        mask = valid & (np.abs(throttle - target) < 40.0)
        if int(np.sum(mask)) < 10:
            continue
        med = float(np.median(gps_speed[mask]))
        spread = float(np.quantile(gps_speed[mask], 0.75) - np.quantile(gps_speed[mask], 0.25))
        results.append((target, int(np.sum(mask)), med, spread))
    return results


def _fused_speed_plateaus(rows: list[dict]) -> list[tuple[int, int, float, float]]:
    if "fused_x" not in rows[0] or "fused_y" not in rows[0]:
        return []
    t = np.asarray([row["_t"] for row in rows], dtype=float)
    steering = np.abs(numeric_array(rows, "steering"))
    throttle = numeric_array(rows, "throttle")
    fix = text_array(rows, "fix")
    x = moving_average(numeric_array(rows, "fused_x"), 11)
    y = moving_average(numeric_array(rows, "fused_y"), 11)
    vx = np.gradient(x, t)
    vy = np.gradient(y, t)
    speed = moving_average(np.hypot(vx, vy), 11)
    accel = np.gradient(speed, t)
    throttle_s = moving_average(throttle, 11)
    throttle_rate = np.gradient(throttle_s, t)
    valid = (
        np.isin(fix, ["RTK FIXED"])
        & (speed > 0.5)
        & (speed < 6.0)
        & (steering < 0.15)
        & (np.abs(accel) < 1.5)
        & (np.abs(throttle_rate) < 250.0)
    )
    results = []
    for target in sorted({int(round(x / 50.0) * 50) for x in throttle[np.isfinite(throttle)]}):
        mask = valid & (np.abs(throttle - target) < 40.0)
        if int(np.sum(mask)) < 50:
            continue
        med = float(np.median(speed[mask]))
        spread = float(np.quantile(speed[mask], 0.75) - np.quantile(speed[mask], 0.25))
        results.append((target, int(np.sum(mask)), med, spread))
    return results


def fit_speed_model(log_paths: list[Path]) -> dict:
    aggregates: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
    for path in log_paths:
        rows = load_rows(path)
        for throttle, count, med, spread in _gps_speed_plateaus(rows) + _fused_speed_plateaus(rows):
            aggregates[throttle].append((count, med, spread))

    points = []
    weights = []
    for throttle in sorted(aggregates):
        total = sum(item[0] for item in aggregates[throttle])
        weighted_med = sum(item[0] * item[1] for item in aggregates[throttle]) / float(total)
        weighted_spread = sum(item[0] * item[2] for item in aggregates[throttle]) / float(total)
        points.append((float(throttle), float(weighted_med), float(weighted_spread)))
        weights.append(float(total))

    data = np.asarray(points, dtype=float)
    weights_arr = np.asarray(weights, dtype=float)
    beta, r2 = weighted_linear_fit(data[:, 0], data[:, 1], weights_arr)
    residual = np.abs((beta[0] + beta[1] * data[:, 0]) - data[:, 1])
    keep = residual < 0.6
    beta, r2 = weighted_linear_fit(data[keep, 0], data[keep, 1], weights_arr[keep])
    return {
        "speed_intercept_mps": float(beta[0]),
        "speed_slope_mps_per_erpm": float(beta[1]),
        "speed_fit_r2": float(r2),
        "speed_plateaus": [
            {
                "throttle": int(data[i, 0]),
                "speed_mps": float(data[i, 1]),
                "spread_mps": float(data[i, 2]),
                "weight": int(weights_arr[i]),
                "kept": bool(keep[i]),
            }
            for i in range(len(data))
        ],
    }


def _delayed_curvature_samples(rows: list[dict], delay_s: float) -> tuple[np.ndarray, np.ndarray]:
    if "fused_x" not in rows[0] or "fused_y" not in rows[0] or "fused_yaw" not in rows[0]:
        return np.asarray([]), np.asarray([])
    t = np.asarray([row["_t"] for row in rows], dtype=float)
    steering = numeric_array(rows, "steering")
    x = moving_average(numeric_array(rows, "fused_x"), 21)
    y = moving_average(numeric_array(rows, "fused_y"), 21)
    yaw = np.unwrap(np.deg2rad(moving_average(numeric_array(rows, "fused_yaw"), 21)))
    fix = text_array(rows, "fix")

    vx = np.gradient(x, t)
    vy = np.gradient(y, t)
    speed = moving_average(np.hypot(vx, vy), 11)
    yaw_rate = moving_average(np.gradient(yaw, t), 11)
    curvature = np.divide(yaw_rate, speed, out=np.full_like(yaw_rate, np.nan), where=speed > 0.5)
    steering_delayed = np.interp(t - delay_s, t, steering, left=np.nan, right=np.nan)
    valid = (
        np.isin(fix, ["RTK FIXED"])
        & np.isfinite(curvature)
        & np.isfinite(steering_delayed)
        & (speed > 0.8)
        & (speed < 5.0)
        & (np.abs(curvature) < 0.8)
        & (np.abs(steering_delayed) < 0.95)
    )
    return steering_delayed[valid], curvature[valid]


def _binned_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, list[tuple[float, float, int]]]:
    edges = np.linspace(-0.9, 0.9, 13)
    x_bins = []
    y_bins = []
    counts = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (x >= lo) & (x < hi)
        if int(np.sum(mask)) < 120:
            continue
        x_bins.append(float(np.median(x[mask])))
        y_bins.append(float(np.median(y[mask])))
        counts.append(int(np.sum(mask)))
    x_arr = np.asarray(x_bins, dtype=float)
    y_arr = np.asarray(y_bins, dtype=float)
    w_arr = np.asarray(counts, dtype=float)
    beta, r2 = weighted_linear_fit(x_arr, y_arr, w_arr)
    return beta, float(r2), list(zip(x_bins, y_bins, counts))


def fit_steering_model(log_paths: list[Path]) -> dict:
    steering_paths = [path for path in log_paths if path.name in {"data9.csv", "data011.csv"}]
    best = None
    best_bins = None
    for delay_s in np.arange(0.20, 0.81, 0.02):
        xs = []
        ys = []
        for path in steering_paths:
            rows = load_rows(path)
            s, c = _delayed_curvature_samples(rows, float(delay_s))
            if len(s):
                xs.append(s)
                ys.append(c)
        if not xs:
            continue
        x_all = np.concatenate(xs)
        y_all = np.concatenate(ys)
        beta, r2, bins = _binned_fit(x_all, y_all)
        if best is None or r2 > best[2]:
            best = (float(delay_s), beta, float(r2))
            best_bins = bins

    if best is None:
        raise RuntimeError("Could not fit steering model from logs.")
    delay_s, beta, r2 = best
    return {
        "steering_delay_s": delay_s,
        "curvature_intercept_inv_m": float(beta[0]),
        "curvature_gain_inv_m": float(beta[1]),
        "steering_trim": float(-beta[0] / beta[1]),
        "steering_fit_r2": float(r2),
        "steering_bins": [
            {"steering": float(s), "curvature_inv_m": float(c), "count": int(n)} for s, c, n in (best_bins or [])
        ],
    }


def fit_sample_rates(log_paths: list[Path]) -> dict:
    drive_dts = []
    gps_dts = []
    for path in log_paths:
        rows = load_rows(path)
        t = np.asarray([row["_t"] for row in rows], dtype=float)
        if len(t) > 1:
            drive_dts.extend(np.diff(t))

        last_lat = None
        last_lon = None
        last_t = None
        for row in rows:
            lat = row.get("latitude")
            lon = row.get("longitude")
            if lat in ("", None) or lon in ("", None):
                continue
            lat = float(lat)
            lon = float(lon)
            current_t = float(row["_t"])
            if last_lat is None or abs(lat - last_lat) > 1e-12 or abs(lon - last_lon) > 1e-12:
                if last_t is not None:
                    gps_dts.append(current_t - last_t)
                last_t = current_t
                last_lat = lat
                last_lon = lon

    drive_loop_hz = 1.0 / float(np.median(drive_dts))
    gps_rate_hz = 1.0 / float(np.median(gps_dts)) if gps_dts else 10.0
    return {
        "drive_loop_hz": float(round(drive_loop_hz)),
        "gps_rate_hz": float(round(gps_rate_hz)),
        "imu_rate_hz": float(round(drive_loop_hz)),
        "sim_rate_hz": float(max(200.0, round(drive_loop_hz) * 4.0)),
    }


def fit_sensor_noise(log_paths: list[Path]) -> dict:
    gps_sigma: dict[str, list[float]] = defaultdict(list)
    speed_errors = []
    imu_heading_errors = []
    imu_acc = []

    for path in log_paths:
        rows = load_rows(path)
        if not rows:
            continue
        valid_latlon = [row for row in rows if row.get("latitude") not in ("", None) and row.get("longitude") not in ("", None)]
        if not valid_latlon:
            continue
        ref_lat = float(valid_latlon[0]["latitude"])
        ref_lon = float(valid_latlon[0]["longitude"])
        cos_ref = math.cos(math.radians(ref_lat))
        earth = 6378137.0

        t = np.asarray([row["_t"] for row in rows], dtype=float)
        if "fused_x" in rows[0] and "fused_y" in rows[0]:
            fx = numeric_array(rows, "fused_x")
            fy = numeric_array(rows, "fused_y")
            vx = np.gradient(moving_average(fx, 11), t)
            vy = np.gradient(moving_average(fy, 11), t)
            fused_speed = moving_average(np.hypot(vx, vy), 11)
        else:
            fused_speed = np.full(len(rows), np.nan)

        for idx, row in enumerate(rows):
            lat = row.get("latitude")
            lon = row.get("longitude")
            fix = row.get("fix", "")
            if lat not in ("", None) and lon not in ("", None) and row.get("fused_x", "") not in ("", None) and row.get("fused_y", "") not in ("", None):
                lat = float(lat)
                lon = float(lon)
                raw_x = math.radians(lon - ref_lon) * cos_ref * earth
                raw_y = math.radians(lat - ref_lat) * earth
                err = math.hypot(raw_x - float(row["fused_x"]), raw_y - float(row["fused_y"]))
                gps_sigma[fix].append(err)

            if row.get("gps_speed", "") not in ("", None) and np.isfinite(fused_speed[idx]):
                speed_errors.append(float(row["gps_speed"]) - float(fused_speed[idx]))

            if row.get("imu_heading", "") not in ("", None) and row.get("fused_yaw", "") not in ("", None):
                heading = float(row["imu_heading"])
                fused_yaw = float(row["fused_yaw"])
                imu_yaw = (90.0 - heading + 180.0) % 360.0 - 180.0
                diff = ((imu_yaw - fused_yaw + 180.0) % 360.0) - 180.0
                imu_heading_errors.append(diff)

            if row.get("imu_accuracy_deg", "") not in ("", None):
                imu_acc.append(float(row["imu_accuracy_deg"]))

    gps_sigma_out = {
        key: float(max(0.02, np.median(values))) if values else default
        for key, default in {"RTK FIXED": 0.02, "RTK FLOAT": 0.35, "3D": 1.0, "NO FIX": 3.0}.items()
        for values in [gps_sigma.get(key, [])]
    }
    gps_sigma_out["NO FIX"] = max(gps_sigma_out.get("NO FIX", 3.0), 3.0)

    imu_acc_arr = np.asarray(imu_acc, dtype=float) if imu_acc else np.asarray([5.5], dtype=float)
    return {
        "gps_position_sigma_m": gps_sigma_out,
        "gps_speed_sigma_mps": float(max(0.03, np.std(speed_errors))) if speed_errors else 0.08,
        "imu_heading_sigma_deg": float(max(0.5, np.std(imu_heading_errors))) if imu_heading_errors else 1.5,
        "imu_accuracy_deg_mean": float(np.mean(imu_acc_arr)),
        "imu_accuracy_deg_std": float(max(0.2, np.std(imu_acc_arr))),
        "imu_accel_sigma_mps2": 0.10,
        "imu_gyro_sigma_dps": 0.35,
    }


def fit_speed_lag(log_paths: list[Path]) -> float:
    taus = []
    for path in log_paths:
        rows = load_rows(path)
        if "fused_x" not in rows[0] or "fused_y" not in rows[0]:
            continue
        t = np.asarray([row["_t"] for row in rows], dtype=float)
        steering = np.abs(numeric_array(rows, "steering"))
        throttle = numeric_array(rows, "throttle")
        fix = text_array(rows, "fix")
        x = moving_average(numeric_array(rows, "fused_x"), 11)
        y = moving_average(numeric_array(rows, "fused_y"), 11)
        vx = np.gradient(x, t)
        vy = np.gradient(y, t)
        speed = moving_average(np.hypot(vx, vy), 11)

        for idx in range(20, len(t) - 60):
            if abs(throttle[idx] - throttle[idx - 1]) < 300.0:
                continue
            if steering[idx] > 0.15 or fix[idx] != "RTK FIXED":
                continue
            before = np.median(speed[max(0, idx - 15):idx])
            after = np.median(speed[idx + 20:min(len(speed), idx + 60)])
            delta = after - before
            if abs(delta) < 0.2:
                continue
            target = before + 0.632 * delta
            segment = speed[idx:min(len(speed), idx + 60)]
            hit = np.where((segment - target) * np.sign(delta) >= 0.0)[0]
            if len(hit) == 0:
                continue
            taus.append(float(t[idx + hit[0]] - t[idx]))
    if not taus:
        return 0.60
    return float(np.clip(np.median(taus), 0.25, 1.25))


def build_payload(log_paths: list[Path]) -> dict:
    speed = fit_speed_model(log_paths)
    steering = fit_steering_model(log_paths)
    rates = fit_sample_rates(log_paths)
    noise = fit_sensor_noise(log_paths)
    speed_lag_s = fit_speed_lag(log_paths)
    raw_gps_sigma = noise["gps_position_sigma_m"]
    effective_gps_sigma = {
        "RTK FIXED": 0.003,
        "RTK FLOAT": 0.08,
        "3D": 0.30,
        "NO FIX": 1.0,
    }

    payload = {
        "version": 1,
        "source_logs": [path.name for path in log_paths],
        "drive_loop_hz": rates["drive_loop_hz"],
        "gps_rate_hz": rates["gps_rate_hz"],
        "imu_rate_hz": rates["imu_rate_hz"],
        "sim_rate_hz": rates["sim_rate_hz"],
        "start_offset_m": 0.35,
        "speed_intercept_mps": speed["speed_intercept_mps"],
        "speed_slope_mps_per_erpm": speed["speed_slope_mps_per_erpm"],
        "speed_lag_s": speed_lag_s,
        "max_speed_mps": 5.0,
        "steering_trim": steering["steering_trim"],
        "curvature_intercept_inv_m": steering["curvature_intercept_inv_m"],
        "curvature_gain_inv_m": steering["curvature_gain_inv_m"],
        "curvature_cubic_inv_m": 0.0,
        "steering_delay_s": steering["steering_delay_s"],
        "steering_lag_s": 0.15,
        "steering_rate_limit_per_s": 4.0,
        "gps_fix_probabilities": {"RTK FIXED": 0.90, "RTK FLOAT": 0.03, "3D": 0.06, "NO FIX": 0.01},
        "gps_position_sigma_m": effective_gps_sigma,
        "gps_noise_correlation_s": 1.5,
        "gps_white_noise_fraction": 0.15,
        "gps_speed_sigma_mps": noise["gps_speed_sigma_mps"],
        "gps_heading_sigma_deg": 3.0,
        "imu_heading_sigma_deg": noise["imu_heading_sigma_deg"],
        "imu_accuracy_deg_mean": noise["imu_accuracy_deg_mean"],
        "imu_accuracy_deg_std": noise["imu_accuracy_deg_std"],
        "imu_accel_sigma_mps2": noise["imu_accel_sigma_mps2"],
        "imu_gyro_sigma_dps": noise["imu_gyro_sigma_dps"],
        "altitude_m": 0.0,
        "startup_warmup_loops": 10,
        "startup_clip_duration_s": 4.0,
        "startup_clip_throttle": 1500.0,
        "fit_report": {
            "speed": speed,
            "steering": steering,
            "speed_lag_s": speed_lag_s,
            "raw_sensor_noise": noise,
            "effective_gps_position_sigma_m": effective_gps_sigma,
        },
    }
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit kart simulation parameters from allowed aks logs.")
    parser.add_argument("--aks-dir", default="aks", help="directory containing the data*.csv logs")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="output JSON parameter file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    aks_dir = Path(args.aks_dir).expanduser().resolve()
    log_paths = allowed_logs(aks_dir)
    if not log_paths:
        raise RuntimeError(f"No eligible logs found in {aks_dir}")
    payload = build_payload(log_paths)
    save_kart_params(args.output, payload)
    print(json.dumps(payload["fit_report"], indent=2, sort_keys=True))
    print(f"\nSaved calibrated kart parameters to {args.output}")


if __name__ == "__main__":
    main()

