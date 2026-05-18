from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim.track_loader import load_track_data


def parse_timestamp(value: str) -> float:
    return datetime.strptime(value, "%Y-%m-%d-%H-%M-%S.%f").timestamp()


def read_logger_csv(path: str | Path) -> list[dict]:
    path = Path(path)
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            timestamp = row.get("timestamp")
            if timestamp:
                try:
                    row["_t"] = parse_timestamp(timestamp)
                except ValueError:
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


def _compute_fused_speed(rows: list[dict]) -> np.ndarray:
    if not rows or "fused_x" not in rows[0] or "fused_y" not in rows[0]:
        return np.asarray([])
    t = np.asarray([row["_t"] for row in rows], dtype=float)
    x = numeric_array(rows, "fused_x")
    y = numeric_array(rows, "fused_y")
    if len(t) < 2:
        return np.asarray([])
    kernel = np.ones(11, dtype=float) / 11.0
    x_s = np.convolve(np.pad(x, (5, 5), mode="edge"), kernel, mode="valid")
    y_s = np.convolve(np.pad(y, (5, 5), mode="edge"), kernel, mode="valid")
    vx = np.gradient(x_s, t)
    vy = np.gradient(y_s, t)
    speed = np.hypot(vx, vy)
    return np.where(speed < 10.0, speed, np.nan)


def _point_to_segment_distance(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay
    denom = abx * abx + aby * aby
    if denom <= 1e-12:
        return math.hypot(px - ax, py - ay)
    u = max(0.0, min(1.0, (apx * abx + apy * aby) / denom))
    cx = ax + u * abx
    cy = ay + u * aby
    return math.hypot(px - cx, py - cy)


def _nearest_track_index(track, x: float, y: float) -> int:
    dx = track.x_m - float(x)
    dy = track.y_m - float(y)
    return int(np.argmin(dx * dx + dy * dy))


def compute_track_progress(rows: list[dict], track_csv: str | Path) -> tuple[np.ndarray, np.ndarray]:
    track = load_track_data(track_csv, write_controller_csv=False)
    px = numeric_array(rows, "fused_x")
    py = numeric_array(rows, "fused_y")
    progress = np.full(len(rows), np.nan, dtype=float)
    nearest_index = np.full(len(rows), -1, dtype=int)
    for idx in range(len(rows)):
        if not np.isfinite(px[idx]) or not np.isfinite(py[idx]):
            continue
        nearest_index[idx] = _nearest_track_index(track, px[idx], py[idx])
        progress[idx] = float(track.path_s_m[nearest_index[idx]])
    return progress, nearest_index


def lap_window(rows: list[dict], track_csv: str | Path) -> tuple[slice, dict]:
    track = load_track_data(track_csv, write_controller_csv=False)
    px = numeric_array(rows, "fused_x")
    py = numeric_array(rows, "fused_y")
    valid = np.where(np.isfinite(px) & np.isfinite(py))[0]
    if len(valid) == 0:
        return slice(0, len(rows)), {"lap_found": False, "path_length_m": float(track.path_length_m)}

    points = np.column_stack((px[valid], py[valid]))
    step = np.hypot(np.diff(points[:, 0]), np.diff(points[:, 1]))
    traveled = np.concatenate(([0.0], np.cumsum(step)))
    radius_m = 5.0
    min_distance_m = 0.75 * float(track.path_length_m)

    for start_idx_in_valid in range(0, len(valid), 25):
        start = int(valid[start_idx_in_valid])
        start_pt = points[start_idx_in_valid]
        start_distance = traveled[start_idx_in_valid]
        target_distance = start_distance + min_distance_m
        search_start = int(np.searchsorted(traveled, target_distance))
        for j in range(search_start, len(valid)):
            if float(np.hypot(*(points[j] - start_pt))) <= radius_m:
                end = int(valid[j])
                return slice(start, end + 1), {
                    "lap_found": True,
                    "path_length_m": float(track.path_length_m),
                    "distance_covered_m": float(traveled[j] - start_distance),
                    "row_count": int(end - start + 1),
                }

    start = int(valid[0])
    end = int(valid[-1])
    return slice(start, end + 1), {
        "lap_found": False,
        "path_length_m": float(track.path_length_m),
        "distance_covered_m": float(traveled[-1] - traveled[0]) if len(traveled) else 0.0,
        "row_count": int(end - start + 1),
    }


def compute_cross_track_error(rows: list[dict], track_csv: str | Path) -> np.ndarray:
    track = load_track_data(track_csv, write_controller_csv=False)
    px = numeric_array(rows, "fused_x")
    py = numeric_array(rows, "fused_y")
    errors = np.full(len(rows), np.nan, dtype=float)
    for idx in range(len(rows)):
        if not np.isfinite(px[idx]) or not np.isfinite(py[idx]):
            continue
        best = float("inf")
        for seg in range(len(track.x_m) - 1):
            dist = _point_to_segment_distance(
                float(px[idx]),
                float(py[idx]),
                float(track.x_m[seg]),
                float(track.y_m[seg]),
                float(track.x_m[seg + 1]),
                float(track.y_m[seg + 1]),
            )
            if dist < best:
                best = dist
        errors[idx] = best
    return errors


def summarize_logger_csv(path: str | Path, track_csv: str | Path | None = None) -> dict:
    rows = read_logger_csv(path)
    window_meta = {}
    if track_csv is not None and rows:
        row_slice, window_meta = lap_window(rows, track_csv)
        rows = rows[row_slice]
    gps_speed = numeric_array(rows, "gps_speed")
    steering = numeric_array(rows, "steering")
    fused_speed = _compute_fused_speed(rows)

    summary = {
        "rows": len(rows),
        "gps_speed_mean_mps": float(np.nanmean(gps_speed)) if np.isfinite(gps_speed).any() else None,
        "gps_speed_p95_mps": float(np.nanpercentile(gps_speed, 95.0)) if np.isfinite(gps_speed).any() else None,
        "fused_speed_mean_mps": float(np.nanmean(fused_speed)) if len(fused_speed) else None,
        "fused_speed_p95_mps": float(np.nanpercentile(fused_speed, 95.0)) if len(fused_speed) else None,
        "abs_steering_mean": float(np.nanmean(np.abs(steering))) if np.isfinite(steering).any() else None,
    }
    summary.update(window_meta)
    if track_csv is not None:
        cte = compute_cross_track_error(rows, track_csv)
        if np.isfinite(cte).any():
            summary["cte_mean_m"] = float(np.nanmean(cte))
            summary["cte_p95_m"] = float(np.nanpercentile(cte, 95.0))
            summary["cte_max_m"] = float(np.nanmax(cte))
    return summary


def compare_summaries(real_summary: dict, sim_summary: dict) -> dict:
    metrics = sorted(set(real_summary.keys()) | set(sim_summary.keys()))
    diff = {}
    for key in metrics:
        rv = real_summary.get(key)
        sv = sim_summary.get(key)
        if isinstance(rv, (int, float)) and isinstance(sv, (int, float)):
            diff[key] = {
                "real": rv,
                "sim": sv,
                "abs_diff": abs(float(rv) - float(sv)),
            }
        else:
            diff[key] = {"real": rv, "sim": sv}
    return diff


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare a real kart logger CSV to a simulated logger CSV.")
    parser.add_argument("real_log")
    parser.add_argument("sim_log")
    parser.add_argument("--track", default=None, help="optional waypoint CSV for cross-track error metrics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    real_summary = summarize_logger_csv(args.real_log, args.track)
    sim_summary = summarize_logger_csv(args.sim_log, args.track)
    diff = compare_summaries(real_summary, sim_summary)
    print(json.dumps(diff, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

