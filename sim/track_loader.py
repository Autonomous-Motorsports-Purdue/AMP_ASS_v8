from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from parts.gps_to_xy import GPS_to_xy


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


def _load_numeric_csv(csv_path: Path, skip_header: int = 0) -> np.ndarray:
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


def _load_latlon_points(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    structured = _load_structured_csv(csv_path)
    if structured is not None:
        lat_col = _find_column(structured.dtype.names, ("latitude", "lat", "lat_deg"))
        lon_col = _find_column(structured.dtype.names, ("longitude", "lon", "lon_deg"))
        if lat_col is not None and lon_col is not None:
            return np.asarray(structured[lat_col], dtype=float), np.asarray(structured[lon_col], dtype=float)

    skip_header = 1 if _looks_like_header(_first_data_line(csv_path)) else 0
    data = _load_numeric_csv(csv_path, skip_header=skip_header)
    return np.asarray(data[:, 0], dtype=float), np.asarray(data[:, 1], dtype=float)


def _load_xy_points(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    structured = _load_structured_csv(csv_path)
    if structured is not None:
        x_col = _find_column(structured.dtype.names, ("x_m", "x"))
        y_col = _find_column(structured.dtype.names, ("y_m", "y"))
        if x_col is not None and y_col is not None:
            return np.asarray(structured[x_col], dtype=float), np.asarray(structured[y_col], dtype=float)

    skip_header = 1 if _looks_like_header(_first_data_line(csv_path)) else 0
    data = _load_numeric_csv(csv_path, skip_header=skip_header)
    return np.asarray(data[:, 0], dtype=float), np.asarray(data[:, 1], dtype=float)


def build_path_s(x_m: np.ndarray, y_m: np.ndarray) -> np.ndarray:
    segment_lengths = np.hypot(np.diff(x_m), np.diff(y_m))
    return np.concatenate(([0.0], np.cumsum(segment_lengths)))


def build_path_heading(x_m: np.ndarray, y_m: np.ndarray) -> np.ndarray:
    if len(x_m) < 2:
        return np.zeros_like(x_m)
    heading = np.arctan2(np.diff(y_m), np.diff(x_m))
    heading = np.unwrap(heading)
    return np.concatenate((heading, [heading[-1]]))


def average_heading(x_m: np.ndarray, y_m: np.ndarray, n_segments: int = 5) -> float:
    segment_count = min(max(1, len(x_m) - 1), n_segments)
    headings = [math.atan2(y_m[i + 1] - y_m[i], x_m[i + 1] - x_m[i]) for i in range(segment_count)]
    return math.atan2(sum(math.sin(h) for h in headings), sum(math.cos(h) for h in headings))


def _controller_copy_path(source_csv: Path, suffix: str) -> Path:
    return source_csv.with_name(f"{source_csv.stem}{suffix}{source_csv.suffix}")


def _drop_duplicate_loop_endpoint(x_m: np.ndarray, y_m: np.ndarray, lat=None, lon=None):
    if len(x_m) >= 2 and float(np.hypot(x_m[-1] - x_m[0], y_m[-1] - y_m[0])) <= 1e-6:
        x_m = x_m[:-1]
        y_m = y_m[:-1]
        if lat is not None:
            lat = lat[:-1]
        if lon is not None:
            lon = lon[:-1]
    return x_m, y_m, lat, lon


def _write_xy_copy(output_csv: Path, x_m: np.ndarray, y_m: np.ndarray, psi_rad: np.ndarray) -> Path:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        output_csv,
        np.column_stack((x_m, y_m, psi_rad)),
        delimiter=",",
        fmt="%.6f",
        header="x_m,y_m,psi_rad",
        comments="",
    )
    return output_csv


def _write_cte_controller_copy(output_csv: Path, x_m: np.ndarray, y_m: np.ndarray, throttle_pwm: float) -> Path:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = np.column_stack((x_m, y_m, np.full(len(x_m), float(throttle_pwm), dtype=float)))
    np.savetxt(output_csv, rows, delimiter=",", fmt="%.6f")
    return output_csv


@dataclass
class TrackData:
    csv_path: Path
    controller_csv: Path
    cte_controller_csv: Path
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
    source_csv: str | Path,
    input_format: str = "auto",
    loop_tolerance_m: float = 2.0,
    reverse_path: bool = False,
    default_throttle_pwm: float = 2500.0,
    write_controller_csv: bool = True,
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
        x_m = np.asarray(x_src, dtype=float)
        y_m = np.asarray(y_src, dtype=float)
        x_m, y_m, lat, lon = _drop_duplicate_loop_endpoint(
            x_m,
            y_m,
            np.asarray(lat, dtype=float),
            np.asarray(lon, dtype=float),
        )
        source_kind = "latlon"
    else:
        lat = None
        lon = None
        gps_frame = None
        x_m, y_m = _load_xy_points(source_csv)
        if reverse_path:
            x_m = x_m[::-1]
            y_m = y_m[::-1]
        x_m, y_m, _, _ = _drop_duplicate_loop_endpoint(x_m, y_m)
        source_kind = "xy"

    psi_rad = build_path_heading(x_m, y_m)
    path_s_m = build_path_s(x_m, y_m)
    path_length_m = float(path_s_m[-1]) if len(path_s_m) else 0.0
    is_loop = bool(np.hypot(x_m[0] - x_m[-1], y_m[0] - y_m[-1]) <= loop_tolerance_m)

    if source_kind == "latlon":
        controller_csv = _controller_copy_path(source_csv, "_xy")
    elif has_header or reverse_path:
        controller_csv = _controller_copy_path(source_csv, "_xy") if reverse_path else source_csv
    else:
        controller_csv = _controller_copy_path(source_csv, "_normalized_xy")

    cte_controller_csv = _controller_copy_path(source_csv, "_xy_throttle")

    if write_controller_csv:
        _write_xy_copy(controller_csv, x_m, y_m, psi_rad)
        _write_cte_controller_copy(cte_controller_csv, x_m, y_m, default_throttle_pwm)

    return TrackData(
        csv_path=source_csv,
        controller_csv=controller_csv,
        cte_controller_csv=cte_controller_csv,
        has_geo=bool(source_kind == "latlon"),
        source_kind=source_kind,
        ref_lat_deg=None if lat is None else float(lat[0]),
        ref_lon_deg=None if lon is None else float(lon[0]),
        lat=None if lat is None else np.asarray(lat, dtype=float),
        lon=None if lon is None else np.asarray(lon, dtype=float),
        x_m=np.asarray(x_m, dtype=float),
        y_m=np.asarray(y_m, dtype=float),
        psi_rad=np.asarray(psi_rad, dtype=float),
        path_s_m=np.asarray(path_s_m, dtype=float),
        path_length_m=path_length_m,
        is_loop=is_loop,
        gps_frame=gps_frame,
    )


def ensure_cte_controller_csv(
    source_csv: str | Path,
    default_throttle_pwm: float = 2500.0,
    input_format: str = "auto",
) -> Path:
    track = load_track_data(
        source_csv=source_csv,
        input_format=input_format,
        default_throttle_pwm=default_throttle_pwm,
        write_controller_csv=True,
    )
    return track.cte_controller_csv

