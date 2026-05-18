#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import animation
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from parts.gps_to_xy import GPS_to_xy

RTK_GPS_NOISE_M = 0.02
FLOAT_GPS_NOISE_M = 0.5
FLOAT_GPS_PROB = 0.1


class PIDController:
    def __init__(self, p=0, i=0, d=0, debug=False, fixed_dt=None):
        self.Kp = p
        self.Ki = i
        self.Kd = d
        self.target = 0
        self.prev_tm = time.time()
        self.prev_error = 0
        self.error = None
        self.alpha = 0
        self.debug = debug
        self.fixed_dt = fixed_dt

    def run(self, target_value, feedback):
        curr_tm = time.time()
        self.target = target_value
        error = self.error = self.target - feedback

        if self.fixed_dt is not None:
            dt = float(self.fixed_dt)
        else:
            dt = curr_tm - self.prev_tm

        curr_alpha = 0
        curr_alpha += self.Kp * error
        curr_alpha += self.Ki * (error * dt)
        if dt > 0:
            # Derivative on error dampens oscillation instead of amplifying it.
            curr_alpha += self.Kd * ((error - self.prev_error) / float(dt))

        self.prev_tm = curr_tm
        self.prev_error = error
        self.alpha = curr_alpha

        if self.debug:
            print("PID target value:", round(target_value, 4))
            print("PID feedback value:", round(feedback, 4))
            print("PID output:", round(curr_alpha, 4))

        return curr_alpha


class CTE(object):
    def __init__(self, look_ahead=1, look_behind=1, num_pts=None, geometry_mode: str = "secant") -> None:
        self.num_pts = num_pts
        self.look_ahead = look_ahead
        self.look_behind = look_behind
        self.geometry_mode = geometry_mode

    def nearest_pt(self, path, x, y, from_pt=0, num_pts=None):
        from_pt = from_pt if from_pt is not None else 0
        num_pts = num_pts if num_pts is not None else len(path)
        num_pts = min(num_pts, len(path))
        if num_pts < 0:
            logging.error("num_pts must not be negative.")
            return None, None, None

        min_pt = None
        min_dist = None
        min_index = None
        for j in range(num_pts):
            i = (j + from_pt) % len(path)
            p = path[i]
            d = math.hypot(p[0] - x, p[1] - y)
            if min_dist is None or d < min_dist:
                min_pt = p
                min_dist = d
                min_index = i
        return min_pt, min_index, min_dist

    def nearest_two_pts(self, path, x, y):
        if path is None or len(path) < 2:
            logging.error("path is none; cannot calculate nearest points")
            return None, None

        distances = []
        for iP, p in enumerate(path):
            d = math.hypot(p[0] - x, p[1] - y)
            distances.append((d, iP, p))
        distances.sort(key=lambda elem: elem[0])

        iA = (distances[0][1] - 1) % len(path)
        a = path[iA]
        iB = (iA + 2) % len(path)
        b = path[iB]

        return a, b

    def nearest_waypoints(self, path, x, y, look_ahead=1, look_behind=1, from_pt=0, num_pts=None):
        if path is None or len(path) < 2:
            logging.error("path is none; cannot calculate nearest points")
            return None, None

        if look_ahead < 0:
            logging.error("look_ahead must be a non-negative number")
            return None, None
        if look_behind < 0:
            logging.error("look_behind must be a non-negative number")
            return None, None
        if (look_ahead + look_behind) > len(path):
            logging.error("the path is not long enough to supply the waypoints")
            return None, None

        _pt, i, _distance = self.nearest_pt(path, x, y, from_pt, num_pts)
        a = (i + len(path) - look_behind) % len(path)
        b = (i + look_ahead) % len(path)
        return a, i, b

    def nearest_track(self, path, x, y, look_ahead=1, look_behind=1, from_pt=0, num_pts=None):
        a, i, b = self.nearest_waypoints(path, x, y, look_ahead, look_behind, from_pt, num_pts)
        return (path[a], path[b], i) if a is not None and b is not None else (None, None, None)

    def run(self, path, x, y, from_pt=None):
        cte = 0.0
        i = from_pt
        a, b, i = self.nearest_track(
            path,
            x,
            y,
            look_ahead=self.look_ahead,
            look_behind=self.look_behind,
            from_pt=from_pt,
            num_pts=self.num_pts,
        )

        if a is not None and b is not None:
            logging.info(f"nearest: ({a[0]}, {a[1]}) to ({x}, {y})")
            if self.geometry_mode == "rk4":
                # Fit a quadratic through a point behind, a point in the middle, and a point ahead.
                a_idx, i_idx, b_idx = self.nearest_waypoints(
                    path,
                    x,
                    y,
                    look_ahead=self.look_ahead,
                    look_behind=self.look_behind,
                    from_pt=from_pt,
                    num_pts=self.num_pts,
                )
                if a_idx is None or b_idx is None:
                    return cte, i
                mid_idx = (i_idx + max(1, self.look_ahead // 2)) % len(path)
                p0 = np.asarray(path[a_idx][:2], dtype=float)
                p1 = np.asarray(path[mid_idx][:2], dtype=float)
                p2 = np.asarray(path[b_idx][:2], dtype=float)
                curve_samples = _fit_quadratic_curve_samples(p0, p1, p2, steps=32)
                vehicle = np.asarray([x, y], dtype=float)
                dists = np.linalg.norm(curve_samples - vehicle, axis=1)
                closest_idx = int(np.argmin(dists))
                closest_pt = curve_samples[closest_idx]
                if closest_idx < len(curve_samples) - 1:
                    tangent = curve_samples[closest_idx + 1] - closest_pt
                else:
                    tangent = closest_pt - curve_samples[closest_idx - 1]
                tangent_norm = float(np.linalg.norm(tangent))
                if tangent_norm > 1e-9:
                    cross_z = tangent[0] * (y - closest_pt[1]) - tangent[1] * (x - closest_pt[0])
                    cte = -cross_z / tangent_norm
            else:
                dx = b[0] - a[0]
                dy = b[1] - a[1]
                seg_len = math.hypot(dx, dy)
                if seg_len > 1e-9:
                    # Positive cross product means the vehicle is left of the track direction.
                    cross_z = dx * (y - a[1]) - dy * (x - a[0])
                    cte = -cross_z / seg_len
        else:
            logging.info(f"no nearest point to ({x},{y}))")
        return cte, i


class CTEController:
    def __init__(
        self,
        path_csv,
        throttle=1000,
        kp=0.2,
        ki=0.005,
        kd=0.1,
        look_ahead=5,
        look_behind=1,
        geometry_mode: str = "secant",
        fixed_dt=None,
    ):
        self.cte = CTE(look_ahead=look_ahead, look_behind=look_behind, geometry_mode=geometry_mode)
        self.pid = PIDController(p=kp, i=ki, d=kd, debug=False, fixed_dt=fixed_dt)
        self.path = np.atleast_2d(
            np.genfromtxt(path_csv, delimiter=",", skip_header=1, dtype=float, encoding="utf-8")
        )
        if self.path.shape[0] < 2:
            raise ValueError(f"Path must contain at least 2 waypoints: {path_csv}")
        self.throttle = float(throttle)
        self.geometry_mode = geometry_mode

    def run(self, x, y, yaw):
        cte, idx = self.cte.run(self.path, x, y)
        steer = self.pid.run(0.0, cte)
        if self.pid.debug:
            print("CTE:", round(cte, 4))
        return self.throttle, steer


def normalize_angle(angle_rad):
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def _norm_name(name: str) -> str:
    return name.lower().replace(" ", "").replace("_", "")


def _first_data_line(csv_path: Path) -> str:
    with csv_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                return stripped
    return ""


def _looks_like_header(first_line: str) -> bool:
    return any(ch.isalpha() for ch in first_line)


def _load_structured_csv(csv_path: Path):
    first_line = _first_data_line(csv_path)
    if not _looks_like_header(first_line):
        return None
    data = np.genfromtxt(
        csv_path,
        delimiter=",",
        names=True,
        comments="#",
        autostrip=True,
        dtype=float,
        encoding="utf-8",
    )
    return data if data.dtype.names is not None else None


def _find_column(dtype_names, candidates):
    lower_map = {_norm_name(name): name for name in dtype_names}
    for candidate in candidates:
        key = _norm_name(candidate)
        if key in lower_map:
            return lower_map[key]
    return None


def _load_numeric_csv(csv_path: Path, skip_header: int = 0):
    data = np.genfromtxt(
        csv_path,
        delimiter=",",
        comments="#",
        dtype=float,
        encoding="utf-8",
        skip_header=skip_header,
    )
    data = np.atleast_2d(data)
    if data.shape[1] < 2:
        raise ValueError(f"Expected at least 2 columns in {csv_path}, found {data.shape[1]}.")
    return data


def _load_latlon_points(csv_path: Path):
    structured = _load_structured_csv(csv_path)
    if structured is not None:
        lat_col = _find_column(structured.dtype.names, ("latitude", "lat", "lat_deg"))
        lon_col = _find_column(structured.dtype.names, ("longitude", "lon", "lon_deg"))
        if lat_col is not None and lon_col is not None:
            return np.asarray(structured[lat_col], dtype=float), np.asarray(structured[lon_col], dtype=float)

    skip_header = 1 if _looks_like_header(_first_data_line(csv_path)) else 0
    data = _load_numeric_csv(csv_path, skip_header=skip_header)
    return np.asarray(data[:, 0], dtype=float), np.asarray(data[:, 1], dtype=float)


def _load_xy_points(csv_path: Path):
    structured = _load_structured_csv(csv_path)
    if structured is not None:
        x_col = _find_column(structured.dtype.names, ("x_m", "x"))
        y_col = _find_column(structured.dtype.names, ("y_m", "y"))
        if x_col is not None and y_col is not None:
            return np.asarray(structured[x_col], dtype=float), np.asarray(structured[y_col], dtype=float)

    skip_header = 1 if _looks_like_header(_first_data_line(csv_path)) else 0
    data = _load_numeric_csv(csv_path, skip_header=skip_header)
    return np.asarray(data[:, 0], dtype=float), np.asarray(data[:, 1], dtype=float)


def _build_path_s(x_m: np.ndarray, y_m: np.ndarray) -> np.ndarray:
    segment_lengths = np.hypot(np.diff(x_m), np.diff(y_m))
    return np.concatenate(([0.0], np.cumsum(segment_lengths)))


def _build_path_heading(x_m: np.ndarray, y_m: np.ndarray) -> np.ndarray:
    if len(x_m) < 2:
        return np.zeros_like(x_m)
    heading = np.arctan2(np.diff(y_m), np.diff(x_m))
    heading = np.unwrap(heading)
    return np.concatenate((heading, [heading[-1]]))


def _average_heading(x_m: np.ndarray, y_m: np.ndarray, n_segments: int = 5) -> float:
    segment_count = min(max(1, len(x_m) - 1), n_segments)
    headings = [math.atan2(y_m[i + 1] - y_m[i], x_m[i + 1] - x_m[i]) for i in range(segment_count)]
    return math.atan2(sum(math.sin(h) for h in headings), sum(math.cos(h) for h in headings))


def _sample_gps_sigma(rng: np.random.Generator) -> tuple[float, str]:
    if rng.random() < FLOAT_GPS_PROB:
        return FLOAT_GPS_NOISE_M, "float"
    return RTK_GPS_NOISE_M, "rtk"


def _rate_limit_value(current: float, target: float, max_delta: float) -> float:
    delta = float(np.clip(target - current, -max_delta, max_delta))
    return current + delta


def _fit_quadratic_curve_samples(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, steps: int = 32) -> np.ndarray:
    # Fit x(u) and y(u) with a quadratic through three spaced waypoints.
    u_pts = np.array([0.0, 0.5, 1.0], dtype=float)
    a_mat = np.column_stack((u_pts * u_pts, u_pts, np.ones_like(u_pts)))
    coeff_x = np.linalg.solve(a_mat, np.array([p0[0], p1[0], p2[0]], dtype=float))
    coeff_y = np.linalg.solve(a_mat, np.array([p0[1], p1[1], p2[1]], dtype=float))

    steps = max(2, int(steps))
    u_vals = np.linspace(0.0, 1.0, num=steps + 1)
    x_vals = coeff_x[0] * u_vals * u_vals + coeff_x[1] * u_vals + coeff_x[2]
    y_vals = coeff_y[0] * u_vals * u_vals + coeff_y[1] * u_vals + coeff_y[2]
    return np.column_stack((x_vals, y_vals))


def _controller_copy_path(source_csv: Path, normalized: bool = False, reversed_path: bool = False) -> Path:
    suffix = "_reversed" if reversed_path else ""
    if normalized:
        if source_csv.stem.endswith("_xy"):
            return source_csv.with_name(f"{source_csv.stem}{suffix}_normalized{source_csv.suffix}")
        return source_csv.with_name(f"{source_csv.stem}{suffix}_xy{source_csv.suffix}")
    return source_csv.with_name(f"{source_csv.stem}{suffix}_xy{source_csv.suffix}")


def _write_xy_copy(output_csv: Path, x_m: np.ndarray, y_m: np.ndarray, psi_rad: np.ndarray, overwrite: bool) -> Path:
    if output_csv.exists() and not overwrite:
        return output_csv

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        output_csv,
        np.column_stack((x_m, y_m, psi_rad)),
        delimiter=",",
        fmt="%.6f",
        header="x_m, y_m, psi_rad",
        comments="",
    )
    return output_csv


@dataclass
class TrackData:
    csv_path: Path
    controller_csv: Path
    has_geo: bool
    source_kind: str
    ref_lat_deg: float | None
    ref_lon_deg: float | None
    lat: np.ndarray | None
    lon: np.ndarray | None
    x_m: np.ndarray
    y_m: np.ndarray
    psi_rad: np.ndarray
    path_s_m: np.ndarray
    path_length_m: float
    is_loop: bool
    gps_frame: GPS_to_xy | None


def load_track_data(
    source_csv: Path,
    input_format: str = "auto",
    overwrite_xy: bool = False,
    loop_tolerance_m: float = 2.0,
    reverse_path: bool = False,
) -> TrackData:
    source_csv = Path(source_csv).expanduser().resolve()
    structured = _load_structured_csv(source_csv)
    has_header = structured is not None

    has_latlon = False
    if structured is not None:
        has_latlon = (
            _find_column(structured.dtype.names, ("latitude", "lat", "lat_deg")) is not None
            and _find_column(structured.dtype.names, ("longitude", "lon", "lon_deg")) is not None
        )

    use_latlon = input_format == "latlon" or (input_format == "auto" and has_latlon)
    if use_latlon:
        lat, lon = _load_latlon_points(source_csv)
        if reverse_path:
            lat = lat[::-1]
            lon = lon[::-1]

        gps_frame = GPS_to_xy(ref_lat_deg=float(lat[0]), ref_lon_deg=float(lon[0]))
        x_src, y_src = gps_frame.to_xy(lat, lon)
        x_src = np.asarray(x_src, dtype=float)
        y_src = np.asarray(y_src, dtype=float)
        psi_src = _build_path_heading(x_src, y_src)
        controller_csv = _controller_copy_path(source_csv, normalized=False, reversed_path=reverse_path)
        controller_csv = _write_xy_copy(controller_csv, x_src, y_src, psi_src, overwrite=overwrite_xy)
        x_m, y_m = _load_xy_points(controller_csv)
        psi_rad = _build_path_heading(x_m, y_m)
        path_s_m = _build_path_s(x_m, y_m)
        path_length_m = float(path_s_m[-1]) if len(path_s_m) else 0.0
        is_loop = bool(np.hypot(x_m[0] - x_m[-1], y_m[0] - y_m[-1]) <= loop_tolerance_m)
        return TrackData(
            csv_path=source_csv,
            controller_csv=controller_csv,
            has_geo=True,
            source_kind="latlon",
            ref_lat_deg=float(lat[0]),
            ref_lon_deg=float(lon[0]),
            lat=np.asarray(lat, dtype=float),
            lon=np.asarray(lon, dtype=float),
            x_m=x_m,
            y_m=y_m,
            psi_rad=psi_rad,
            path_s_m=path_s_m,
            path_length_m=path_length_m,
            is_loop=is_loop,
            gps_frame=gps_frame,
        )

    x_m, y_m = _load_xy_points(source_csv)
    if reverse_path:
        x_m = x_m[::-1]
        y_m = y_m[::-1]
    if has_header:
        controller_csv = _controller_copy_path(source_csv, normalized=False, reversed_path=reverse_path) if reverse_path else source_csv
        if reverse_path:
            psi_src = _build_path_heading(x_m, y_m)
            controller_csv = _write_xy_copy(controller_csv, x_m, y_m, psi_src, overwrite=overwrite_xy)
            x_m, y_m = _load_xy_points(controller_csv)
    else:
        controller_csv = _controller_copy_path(source_csv, normalized=True, reversed_path=reverse_path)
        psi_src = _build_path_heading(x_m, y_m)
        controller_csv = _write_xy_copy(controller_csv, x_m, y_m, psi_src, overwrite=overwrite_xy)
        x_m, y_m = _load_xy_points(controller_csv)

    psi_rad = _build_path_heading(x_m, y_m)
    path_s_m = _build_path_s(x_m, y_m)
    path_length_m = float(path_s_m[-1]) if len(path_s_m) else 0.0
    is_loop = bool(np.hypot(x_m[0] - x_m[-1], y_m[0] - y_m[-1]) <= loop_tolerance_m)
    return TrackData(
        csv_path=source_csv,
        controller_csv=controller_csv,
        has_geo=False,
        source_kind="xy",
        ref_lat_deg=None,
        ref_lon_deg=None,
        lat=None,
        lon=None,
        x_m=x_m,
        y_m=y_m,
        psi_rad=psi_rad,
        path_s_m=path_s_m,
        path_length_m=path_length_m,
        is_loop=is_loop,
        gps_frame=None,
    )


class KinematicBicycleModel:
    def __init__(self, x_m: float, y_m: float, heading_rad: float, wheelbase_m: float, dt_s: float):
        self.x_m = float(x_m)
        self.y_m = float(y_m)
        self.heading_rad = float(heading_rad)
        self.wheelbase_m = float(wheelbase_m)
        self.dt_s = float(dt_s)

    def step(self, speed_mps: float, steering_rad: float) -> float:
        self.x_m += speed_mps * math.cos(self.heading_rad) * self.dt_s
        self.y_m += speed_mps * math.sin(self.heading_rad) * self.dt_s
        yaw_rate_radps = speed_mps / self.wheelbase_m * math.tan(steering_rad)
        self.heading_rad = normalize_angle(self.heading_rad + yaw_rate_radps * self.dt_s)
        return yaw_rate_radps


def _should_stop(track: TrackData, x_m: float, y_m: float, progress_m: float, closest_idx: int, t_s: float, min_run_time_s: float, completion_radius_m: float) -> bool:
    if t_s < min_run_time_s:
        return False

    start_dist = math.hypot(x_m - track.x_m[0], y_m - track.y_m[0])
    end_dist = math.hypot(x_m - track.x_m[-1], y_m - track.y_m[-1])

    if track.is_loop:
        return progress_m > 0.75 * track.path_length_m and start_dist <= completion_radius_m

    return progress_m >= max(track.path_length_m - completion_radius_m, 0.0) and end_dist <= completion_radius_m and closest_idx >= len(track.x_m) - 2


def simulate_cte(track: TrackData, controller: CTEController, args: argparse.Namespace):
    rng = np.random.default_rng(args.seed)

    throttle_to_speed_scale = args.max_speed_mps / max(args.controller_throttle, 1e-6)
    target_speed_mps = max(0.0, float(controller.throttle) * throttle_to_speed_scale)
    initial_speed_mps = float(args.initial_speed_mps) if args.initial_speed_mps is not None else target_speed_mps

    start_heading_rad = _average_heading(track.x_m, track.y_m)
    start_x_m = float(track.x_m[0] - args.start_offset_m * math.cos(start_heading_rad))
    start_y_m = float(track.y_m[0] - args.start_offset_m * math.sin(start_heading_rad))
    vehicle = KinematicBicycleModel(start_x_m, start_y_m, start_heading_rad, args.wheelbase_m, args.dt)

    true_speed_mps = initial_speed_mps
    steering_angle_rad = 0.0
    steering_cmd_applied = 0.0
    true_yaw_rate_radps = 0.0

    gps_x_m = start_x_m
    gps_y_m = start_y_m
    gps_yaw_rad = start_heading_rad
    gps_lat = float("nan")
    gps_lon = float("nan")
    gps_fix_state = "rtk"

    gps_period_s = 1.0 / max(args.gps_rate_hz, 1e-6)
    next_gps_t = 0.0

    max_time_s = args.max_time_s
    if max_time_s is None:
        max_time_s = max(30.0, 2.5 * track.path_length_m / max(max(target_speed_mps, initial_speed_mps), 0.1))

    logs: dict[str, list[float]] = {
        "time_s": [],
        "true_x_m": [],
        "true_y_m": [],
        "gps_x_m": [],
        "gps_y_m": [],
        "true_yaw_rad": [],
        "measured_yaw_rad": [],
        "closest_idx": [],
        "target_x_m": [],
        "target_y_m": [],
        "cross_track_error_m": [],
        "steering_raw": [],
        "steering_cmd": [],
        "steering_cmd_applied": [],
        "steering_angle_deg": [],
        "throttle_cmd": [],
        "target_speed_mps": [],
        "true_speed_mps": [],
        "true_yaw_rate_radps": [],
        "progress_m": [],
        "gps_lat": [],
        "gps_lon": [],
        "gps_fix_state": [],
    }

    stop_reason = "timeout"
    num_steps = int(math.ceil(max_time_s / args.dt))
    for step in range(num_steps):
        t_s = step * args.dt

        while t_s + 1e-9 >= next_gps_t:
            gps_sigma_m, gps_fix_state = _sample_gps_sigma(rng)
            gps_noise_e_m = rng.normal(0.0, gps_sigma_m)
            gps_noise_n_m = rng.normal(0.0, gps_sigma_m)
            gps_x_m = vehicle.x_m + gps_noise_e_m
            gps_y_m = vehicle.y_m + gps_noise_n_m
            if track.has_geo and track.gps_frame is not None:
                gps_lat, gps_lon = track.gps_frame.to_latlon(gps_x_m, gps_y_m)
            else:
                gps_lat = float("nan")
                gps_lon = float("nan")
            gps_yaw_rad = normalize_angle(vehicle.heading_rad + rng.normal(0.0, math.radians(args.yaw_noise_deg)))
            next_gps_t += gps_period_s

        throttle_cmd, steering_raw = controller.run(gps_x_m, gps_y_m, gps_yaw_rad)
        throttle_cmd = float(throttle_cmd)
        steering_raw = float(steering_raw)
        steering_cmd = float(np.clip(steering_raw, -1.0, 1.0))
        max_steering_delta = min(2.0, 2.0 * args.dt / max(args.steering_full_sweep_s, 1e-6))
        steering_cmd_applied = _rate_limit_value(steering_cmd_applied, steering_cmd, max_steering_delta)

        target_speed_cmd_mps = max(0.0, throttle_cmd * throttle_to_speed_scale)
        # The kart steering convention is inverted relative to the bicycle model.
        steering_target_rad = -steering_cmd_applied * math.radians(args.max_steer_deg)

        steering_alpha = min(1.0, args.dt / max(args.steering_lag_s, args.dt))
        steering_angle_rad += (steering_target_rad - steering_angle_rad) * steering_alpha

        speed_alpha = min(1.0, args.dt / max(args.throttle_lag_s, args.dt))
        true_speed_mps += (target_speed_cmd_mps - true_speed_mps) * speed_alpha
        true_speed_mps = max(0.0, true_speed_mps)

        true_yaw_rate_radps = vehicle.step(true_speed_mps, steering_angle_rad)

        cte_m, closest_idx = controller.cte.run(controller.path, vehicle.x_m, vehicle.y_m)
        closest_idx = int(closest_idx if closest_idx is not None else 0)
        target_x_m = float(controller.path[closest_idx, 0])
        target_y_m = float(controller.path[closest_idx, 1])
        progress_m = float(track.path_s_m[closest_idx])

        logs["time_s"].append(float(t_s))
        logs["true_x_m"].append(float(vehicle.x_m))
        logs["true_y_m"].append(float(vehicle.y_m))
        logs["gps_x_m"].append(float(gps_x_m))
        logs["gps_y_m"].append(float(gps_y_m))
        logs["true_yaw_rad"].append(float(vehicle.heading_rad))
        logs["measured_yaw_rad"].append(float(gps_yaw_rad))
        logs["closest_idx"].append(float(closest_idx))
        logs["target_x_m"].append(target_x_m)
        logs["target_y_m"].append(target_y_m)
        logs["cross_track_error_m"].append(float(cte_m))
        logs["steering_raw"].append(float(steering_raw))
        logs["steering_cmd"].append(float(steering_cmd))
        logs["steering_cmd_applied"].append(float(steering_cmd_applied))
        logs["steering_angle_deg"].append(float(math.degrees(steering_angle_rad)))
        logs["throttle_cmd"].append(float(throttle_cmd))
        logs["target_speed_mps"].append(float(target_speed_cmd_mps))
        logs["true_speed_mps"].append(float(true_speed_mps))
        logs["true_yaw_rate_radps"].append(float(true_yaw_rate_radps))
        logs["progress_m"].append(progress_m)
        logs["gps_lat"].append(float(gps_lat))
        logs["gps_lon"].append(float(gps_lon))
        logs["gps_fix_state"].append(1.0 if gps_fix_state == "float" else 0.0)

        if not np.isfinite(
            [
                vehicle.x_m,
                vehicle.y_m,
                vehicle.heading_rad,
                true_speed_mps,
                steering_angle_rad,
                cte_m,
                steering_raw,
                steering_cmd,
                steering_cmd_applied,
            ]
        ).all():
            raise RuntimeError("Pseudo sim produced non-finite values.")

        if _should_stop(track, vehicle.x_m, vehicle.y_m, progress_m, closest_idx, t_s, args.min_run_time_s, args.completion_radius_m):
            stop_reason = "completed"
            break

    log_arrays = {key: np.asarray(values, dtype=float) for key, values in logs.items()}
    if len(log_arrays["time_s"]) == 0:
        raise RuntimeError("Pseudo sim did not produce any samples.")

    abs_cte = np.abs(log_arrays["cross_track_error_m"])
    summary = {
        "completed": float(stop_reason == "completed"),
        "stop_reason": stop_reason,
        "sim_time_s": float(log_arrays["time_s"][-1]),
        "mean_abs_cte_m": float(np.mean(abs_cte)),
        "p95_abs_cte_m": float(np.percentile(abs_cte, 95.0)),
        "max_abs_cte_m": float(np.max(abs_cte)),
        "avg_speed_mps": float(np.mean(log_arrays["true_speed_mps"])),
        "steering_saturation_fraction": float(np.mean(np.abs(log_arrays["steering_cmd"]) >= 0.95)),
        "completion_fraction": float(log_arrays["progress_m"][-1] / max(track.path_length_m, 1e-6)),
        "final_progress_m": float(log_arrays["progress_m"][-1]),
    }

    return log_arrays, summary


def save_logs(logs: dict[str, np.ndarray], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(logs.keys()))
        for row in zip(*[logs[key] for key in logs]):
            writer.writerow(row)


def save_plot(track: TrackData, logs: dict[str, np.ndarray], output_png: Path) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(track.x_m, track.y_m, "k--", linewidth=1.5, label="Reference Path")
    axes[0, 0].plot(logs["true_x_m"], logs["true_y_m"], color="tab:blue", label="True Vehicle")
    gps_mask = np.isfinite(logs["gps_x_m"]) & np.isfinite(logs["gps_y_m"])
    if np.any(gps_mask):
        axes[0, 0].scatter(
            logs["gps_x_m"][gps_mask],
            logs["gps_y_m"][gps_mask],
            s=10,
            alpha=0.25,
            color="tab:orange",
            edgecolors="none",
            label="GPS Samples",
        )
    axes[0, 0].scatter(track.x_m[0], track.y_m[0], color="green", marker="o", s=50, label="Start")
    axes[0, 0].scatter(track.x_m[-1], track.y_m[-1], color="red", marker="x", s=50, label="End")
    axes[0, 0].set_title(f"Track View ({track.source_kind})")
    axes[0, 0].set_xlabel("X East (m)")
    axes[0, 0].set_ylabel("Y North (m)")
    axes[0, 0].axis("equal")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(logs["time_s"], logs["steering_raw"], label="Raw Steering Output")
    axes[0, 1].plot(logs["time_s"], logs["steering_cmd"], label="Clipped Steering")
    axes[0, 1].plot(logs["time_s"], logs["steering_cmd_applied"], label="Rate-Limited Steering")
    axes[0, 1].set_title("Steering Command")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Normalized Command")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].plot(logs["time_s"], logs["cross_track_error_m"], label="Cross-Track Error")
    axes[1, 0].axhline(0.0, color="k", linewidth=0.8)
    axes[1, 0].set_title("Tracking Error")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Error (m)")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].plot(logs["time_s"], logs["true_speed_mps"], label="True Speed")
    axes[1, 1].plot(logs["time_s"], logs["target_speed_mps"], linestyle="--", label="Target Speed")
    axes[1, 1].set_title("Speed Response")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("m/s")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig(output_png, dpi=150)
    plt.close(fig)


def save_animation(
    track: TrackData,
    logs: dict[str, np.ndarray],
    output_mp4: Path,
    frame_stride: int = 1,
    max_frames: int = 360,
    follow_window_m: float = 25.0,
    look_ahead: int = 5,
    look_behind: int = 1,
    cte_geometry: str = "secant",
) -> None:
    output_mp4.parent.mkdir(parents=True, exist_ok=True)

    times = logs["time_s"]
    true_x = logs["true_x_m"]
    true_y = logs["true_y_m"]
    gps_x = logs["gps_x_m"]
    gps_y = logs["gps_y_m"]
    steering_raw = logs["steering_raw"]
    steering_cmd = logs["steering_cmd"]
    steering_cmd_applied = logs["steering_cmd_applied"]
    cte = logs["cross_track_error_m"]
    speed = logs["true_speed_mps"]
    target_speed = logs["target_speed_mps"]

    if len(times) == 0:
        raise ValueError("Cannot animate an empty log.")

    if len(times) <= 1:
        frame_idxs = [0]
    else:
        coarse_stride = max(1, int(frame_stride))
        frame_idxs = list(range(0, len(times), coarse_stride))
        if len(frame_idxs) > max_frames:
            frame_idxs = np.linspace(0, len(times) - 1, num=max_frames, dtype=int).tolist()
        elif frame_idxs[-1] != len(times) - 1:
            frame_idxs.append(len(times) - 1)
    frame_iter = tqdm(frame_idxs, desc=f"Rendering {output_mp4.name}", unit="frame")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_track, ax_steer = axes[0]
    ax_cte, ax_speed = axes[1]

    ax_track.plot(track.x_m, track.y_m, "k--", linewidth=1.2, alpha=0.85, label="Reference Path")
    ax_track.scatter(track.x_m[0], track.y_m[0], color="green", marker="o", s=50, label="Start")
    ax_track.scatter(track.x_m[-1], track.y_m[-1], color="red", marker="x", s=50, label="End")
    true_line, = ax_track.plot([], [], color="tab:blue", linewidth=2.0, label="True Vehicle")
    gps_scatter = ax_track.scatter([], [], s=10, alpha=0.25, color="tab:orange", edgecolors="none", label="GPS Samples")
    current_true = ax_track.scatter([], [], color="tab:blue", s=40, zorder=4)
    current_gps = ax_track.scatter([], [], color="tab:orange", s=35, zorder=4)
    cte_segment_line, = ax_track.plot([], [], color="magenta", linewidth=2.5, alpha=0.9, label="CTE Segment")
    cte_curve_line, = ax_track.plot([], [], color="tab:cyan", linewidth=2.4, alpha=0.95, label="CTE Curve (quadratic)")
    lookahead_point = ax_track.scatter([], [], color="yellow", marker="*", s=140, edgecolors="black", linewidths=0.7, zorder=5, label="Lookahead Point")
    lookbehind_point = ax_track.scatter([], [], color="purple", marker="o", s=40, edgecolors="black", linewidths=0.4, zorder=5, label="Segment Endpoints")
    time_text = ax_track.text(0.02, 0.98, "", transform=ax_track.transAxes, va="top", ha="left", bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
    ax_track.set_title(f"Track View ({track.source_kind})")
    ax_track.set_xlabel("X East (m)")
    ax_track.set_ylabel("Y North (m)")
    ax_track.set_box_aspect(1)
    ax_track.set_autoscale_on(False)
    ax_track.grid(True, alpha=0.3)
    ax_track.legend(fontsize=8, loc="best")

    ax_steer.set_title("Steering Command")
    ax_steer.set_xlabel("Time (s)")
    ax_steer.set_ylabel("Normalized Command")
    ax_steer.grid(True, alpha=0.3)
    ax_steer.set_xlim(times[0], times[-1])
    ax_steer.set_ylim(-1.1, 1.1)
    steering_raw_line, = ax_steer.plot([], [], label="Raw Steering Output")
    steering_cmd_line, = ax_steer.plot([], [], label="Clipped Steering")
    steering_applied_line, = ax_steer.plot([], [], label="Rate-Limited Steering")
    steering_cursor = ax_steer.axvline(times[0], color="k", linestyle=":", linewidth=1.0)
    ax_steer.legend()

    ax_cte.set_title("Tracking Error")
    ax_cte.set_xlabel("Time (s)")
    ax_cte.set_ylabel("Error (m)")
    ax_cte.grid(True, alpha=0.3)
    ax_cte.set_xlim(times[0], times[-1])
    cte_pad = max(1.0, float(np.max(np.abs(cte)) * 1.1))
    ax_cte.set_ylim(-cte_pad, cte_pad)
    cte_line, = ax_cte.plot([], [], label="Cross-Track Error")
    cte_cursor = ax_cte.axvline(times[0], color="k", linestyle=":", linewidth=1.0)
    ax_cte.axhline(0.0, color="k", linewidth=0.8)
    ax_cte.legend()

    ax_speed.set_title("Speed Response")
    ax_speed.set_xlabel("Time (s)")
    ax_speed.set_ylabel("m/s")
    ax_speed.grid(True, alpha=0.3)
    ax_speed.set_xlim(times[0], times[-1])
    speed_min = max(0.0, float(min(np.min(speed), np.min(target_speed)) - 0.5))
    speed_max = float(max(np.max(speed), np.max(target_speed)) + 0.5)
    ax_speed.set_ylim(speed_min, max(speed_max, speed_min + 1.0))
    speed_line, = ax_speed.plot([], [], label="True Speed")
    target_speed_line, = ax_speed.plot([], [], linestyle="--", label="Target Speed")
    speed_cursor = ax_speed.axvline(times[0], color="k", linestyle=":", linewidth=1.0)
    ax_speed.legend()

    def init():
        for line in (true_line, steering_raw_line, steering_cmd_line, steering_applied_line, cte_line, speed_line, target_speed_line, cte_segment_line, cte_curve_line):
            line.set_data([], [])
        gps_scatter.set_offsets(np.empty((0, 2)))
        current_true.set_offsets(np.empty((0, 2)))
        current_gps.set_offsets(np.empty((0, 2)))
        lookahead_point.set_offsets(np.empty((0, 2)))
        lookbehind_point.set_offsets(np.empty((0, 2)))
        for cursor in (steering_cursor, cte_cursor, speed_cursor):
            cursor.set_xdata([times[0], times[0]])
        time_text.set_text("")
        half_window = follow_window_m / 2.0
        ax_track.set_xlim(true_x[0] - half_window, true_x[0] + half_window)
        ax_track.set_ylim(true_y[0] - half_window, true_y[0] + half_window)
        return (
            true_line,
            gps_scatter,
            current_true,
            current_gps,
            cte_segment_line,
            cte_curve_line,
            lookahead_point,
            lookbehind_point,
            time_text,
            steering_raw_line,
            steering_cmd_line,
            steering_applied_line,
            steering_cursor,
            cte_line,
            cte_cursor,
            speed_line,
            target_speed_line,
            speed_cursor,
        )

    def update(frame_idx):
        t = times[frame_idx]
        idx_slice = slice(0, frame_idx + 1)

        true_line.set_data(true_x[idx_slice], true_y[idx_slice])
        current_true.set_offsets(np.array([[true_x[frame_idx], true_y[frame_idx]]]))

        gps_idx = np.where(np.isfinite(gps_x[: frame_idx + 1]) & np.isfinite(gps_y[: frame_idx + 1]))[0]
        if len(gps_idx):
            gps_points = np.column_stack((gps_x[gps_idx], gps_y[gps_idx]))
            gps_scatter.set_offsets(gps_points)
            current_gps.set_offsets(np.array([[gps_x[gps_idx[-1]], gps_y[gps_idx[-1]]]]))
        else:
            gps_scatter.set_offsets(np.empty((0, 2)))
            current_gps.set_offsets(np.empty((0, 2)))

        closest_idx = int(round(float(logs["closest_idx"][frame_idx]))) % len(track.x_m)
        seg_a_idx = (closest_idx - int(look_behind)) % len(track.x_m)
        seg_b_idx = (closest_idx + int(look_ahead)) % len(track.x_m)
        mid_idx = (closest_idx + max(1, int(look_ahead) // 2)) % len(track.x_m)
        seg_a = np.array([track.x_m[seg_a_idx], track.y_m[seg_a_idx]])
        seg_b = np.array([track.x_m[seg_b_idx], track.y_m[seg_b_idx]])
        cte_segment_line.set_data([seg_a[0], seg_b[0]], [seg_a[1], seg_b[1]])
        lookahead_point.set_offsets(np.array([[seg_b[0], seg_b[1]]]))
        lookbehind_point.set_offsets(np.array([[seg_a[0], seg_a[1]]]))
        if cte_geometry == "rk4":
            p_mid = np.array([track.x_m[mid_idx], track.y_m[mid_idx]])
            curve_samples = _fit_quadratic_curve_samples(seg_a, p_mid, seg_b, steps=32)
            cte_curve_line.set_data(curve_samples[:, 0], curve_samples[:, 1])
        else:
            cte_curve_line.set_data([], [])

        steering_raw_line.set_data(times[idx_slice], steering_raw[idx_slice])
        steering_cmd_line.set_data(times[idx_slice], steering_cmd[idx_slice])
        steering_applied_line.set_data(times[idx_slice], steering_cmd_applied[idx_slice])
        steering_cursor.set_xdata([t, t])

        cte_line.set_data(times[idx_slice], cte[idx_slice])
        cte_cursor.set_xdata([t, t])

        speed_line.set_data(times[idx_slice], speed[idx_slice])
        target_speed_line.set_data(times[idx_slice], target_speed[idx_slice])
        speed_cursor.set_xdata([t, t])

        half_window = follow_window_m / 2.0
        ax_track.set_xlim(true_x[frame_idx] - half_window, true_x[frame_idx] + half_window)
        ax_track.set_ylim(true_y[frame_idx] - half_window, true_y[frame_idx] + half_window)

        time_text.set_text(f"t = {t:.1f} s")
        return (
            true_line,
            gps_scatter,
            current_true,
            current_gps,
            cte_segment_line,
            cte_curve_line,
            lookahead_point,
            lookbehind_point,
            time_text,
            steering_raw_line,
            steering_cmd_line,
            steering_applied_line,
            steering_cursor,
            cte_line,
            cte_cursor,
            speed_line,
            target_speed_line,
            speed_cursor,
        )

    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=frame_iter,
        interval=50,
        blit=False,
        repeat=False,
    )

    writer = animation.FFMpegWriter(fps=6, codec="libx264", bitrate=1800)
    anim.save(output_mp4, writer=writer)
    plt.close(fig)


def print_summary(track: TrackData, summary: dict[str, float], output_csv: Path, output_png: Path) -> None:
    print(f"Track: {track.csv_path}")
    print(f"Controller path: {track.controller_csv}")
    print(f"Source kind: {track.source_kind}")
    if track.has_geo:
        print(f"Geo origin: lat={track.ref_lat_deg:.8f}, lon={track.ref_lon_deg:.8f}")
    print(
        "Run: "
        f"completed={'yes' if summary['completed'] else 'no'} "
        f"({summary['stop_reason']}) | sim_time={summary['sim_time_s']:.1f}s | "
        f"avg_speed={summary['avg_speed_mps']:.2f} m/s"
    )
    print(
        "Tracking: "
        f"mean_abs_cte={summary['mean_abs_cte_m']:.2f} m | "
        f"p95_abs_cte={summary['p95_abs_cte_m']:.2f} m | "
        f"max_abs_cte={summary['max_abs_cte_m']:.2f} m"
    )
    print(
        "Actuation: "
        f"steering_saturated={100.0 * summary['steering_saturation_fraction']:.1f}% | "
        f"completion={100.0 * summary['completion_fraction']:.1f}%"
    )
    warnings = []
    if not summary["completed"]:
        warnings.append("track was not completed before the time limit")
    if summary["mean_abs_cte_m"] > 2.0:
        warnings.append("mean cross-track error is above 2 m")
    if summary["steering_saturation_fraction"] > 0.2:
        warnings.append("steering was saturated more than 20% of the run")
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("Warnings: none")
    print(f"Saved log CSV: {output_csv}")
    print(f"Saved plot PNG: {output_png}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a pseudo-simulation of the CTE kart controller.")
    parser.add_argument("track_csv", help="waypoint CSV file")
    parser.add_argument("--input-format", choices=["auto", "latlon", "xy"], default="auto", help="hint for parsing the waypoint CSV")
    parser.add_argument("--dt", type=float, default=0.05, help="simulation timestep in seconds")
    parser.add_argument("--gps-rate-hz", type=float, default=4.0, help="GPS measurement rate in Hz")
    parser.add_argument("--gps-noise-m", type=float, default=0.1, help="GPS position noise in meters")
    parser.add_argument("--yaw-noise-deg", type=float, default=0.1, help="yaw measurement noise in degrees")
    parser.add_argument("--controller-throttle", type=float, default=1000.0, help="raw throttle returned by the controller")
    parser.add_argument("--max-speed-mps", type=float, default=5.5, help="speed corresponding to full controller throttle")
    parser.add_argument("--initial-speed-mps", type=float, default=None, help="initial vehicle speed; defaults to the controller target speed")
    parser.add_argument("--wheelbase-m", type=float, default=1.000506, help="vehicle wheelbase in meters")
    parser.add_argument("--max-steer-deg", type=float, default=35.0, help="maximum steering angle in degrees")
    parser.add_argument(
        "--steering-full-sweep-s",
        type=float,
        default=0.1,
        help="time in seconds for steering to move from -1 to 1 after rate limiting",
    )
    parser.add_argument("--steering-lag-s", type=float, default=0.15, help="first-order steering lag")
    parser.add_argument("--throttle-lag-s", type=float, default=0.5, help="first-order speed response lag")
    parser.add_argument("--start-offset-m", type=float, default=0.6, help="start slightly behind the first waypoint")
    parser.add_argument("--min-run-time-s", type=float, default=5.0, help="minimum time before the completion detector can stop the run")
    parser.add_argument("--completion-radius-m", type=float, default=1.0, help="distance threshold for declaring completion")
    parser.add_argument("--loop-tolerance-m", type=float, default=2.0, help="distance threshold used to detect looped paths")
    parser.add_argument("--max-time-s", type=float, default=200.0, help="hard timeout for the pseudo simulation")
    parser.add_argument("--reverse-path", action="store_true", help="simulate the waypoint path in reverse order")
    parser.add_argument("--cte-geometry", choices=["secant", "rk4"], default="secant", help="path geometry used for CTE calculation")
    parser.add_argument("--kp", type=float, default=0.2, help="CTE PID proportional gain")
    parser.add_argument("--ki", type=float, default=0.005, help="CTE PID integral gain")
    parser.add_argument("--kd", type=float, default=0.1, help="CTE PID derivative gain")
    parser.add_argument("--look-ahead", type=int, default=5, help="CTE lookahead waypoint count")
    parser.add_argument("--look-behind", type=int, default=1, help="CTE lookbehind waypoint count")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--save-dir", default="sim_outputs", help="directory for CSV and PNG outputs")
    parser.add_argument("--overwrite-xy", action="store_true", help="overwrite generated XY sidecars when they already exist")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_csv = Path(args.track_csv).expanduser().resolve()
    track = load_track_data(
        source_csv,
        input_format=args.input_format,
        overwrite_xy=args.overwrite_xy,
        loop_tolerance_m=args.loop_tolerance_m,
        reverse_path=args.reverse_path,
    )

    controller = CTEController(
        str(track.controller_csv),
        throttle=args.controller_throttle,
        kp=args.kp,
        ki=args.ki,
        kd=args.kd,
        look_ahead=args.look_ahead,
        look_behind=args.look_behind,
        geometry_mode=args.cte_geometry,
        fixed_dt=args.dt,
    )

    logs, summary = simulate_cte(track, controller, args)

    base_stem = source_csv.stem[:-3] if source_csv.stem.endswith("_xy") else source_csv.stem
    output_stem = f"{base_stem}_cte_pseudo_sim{'_reversed' if args.reverse_path else ''}"
    save_dir = Path(args.save_dir).expanduser().resolve()
    output_csv = save_dir / f"{output_stem}.csv"
    output_png = save_dir / f"{output_stem}.png"
    output_mp4 = save_dir / f"{output_stem}.mp4"

    save_logs(logs, output_csv)
    save_plot(track, logs, output_png)
    save_animation(
        track,
        logs,
        output_mp4,
        look_ahead=args.look_ahead,
        look_behind=args.look_behind,
        cte_geometry=args.cte_geometry,
    )
    print_summary(track, summary, output_csv, output_png)
    print(f"Saved animation MP4: {output_mp4}")


if __name__ == "__main__":
    main()
