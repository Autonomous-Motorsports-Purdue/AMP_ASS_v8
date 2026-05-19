#!/usr/bin/env python3
"""
Read a waypoint CSV (header row required for column names), append throttle_pwm.

Output is data rows only (no header), matching Donkeycar genfromtxt usage with skip_header=0.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

# --- tuning ---
THROTTLE_STRAIGHT = 3450  # PWM on essentially straight lookahead
THROTTLE_MAX_CURVE = 2150  # PWM when lookahead hits the tightest parts of this track
LOOKAHEAD = 10.0  # meters along the closed polyline ahead of each vertex

# Use percentiles of lookahead kappa on *this* CSV so mild bend ≠ half throttle.
STRAIGHT_KAPPA_QUANTILE = 0.28  # below this → near full THROTTLE_STRAIGHT
CURVE_KAPPA_QUANTILE = 0.75 # above this → near THROTTLE_MAX_CURVE
# > 1 keeps PWM closer to straight until kappa approaches CURVE end of range.
CURVE_SHARPNESS = 1.2 
# --------------

MIN_SEG_LEN = 1e-6


def _norm_bearing_delta(a: float, b: float) -> float:
    return abs(math.atan2(math.sin(b - a), math.cos(b - a)))


def vertex_curvatures(xy: np.ndarray) -> np.ndarray:
    """Unsigned turn rate ~ |Δheading| / mean(edge length) at each vertex (closed)."""
    n = len(xy)
    kappa = np.zeros(n)
    if n < 3:
        return kappa

    for i in range(n):
        im, ip = (i - 1) % n, (i + 1) % n
        p0, p1, p2 = xy[im], xy[i], xy[ip]
        e_in = p1 - p0
        e_out = p2 - p1
        li = float(np.hypot(e_in[0], e_in[1]))
        lo = float(np.hypot(e_out[0], e_out[1]))
        if li < MIN_SEG_LEN or lo < MIN_SEG_LEN:
            continue
        th_in = math.atan2(float(e_in[1]), float(e_in[0]))
        th_out = math.atan2(float(e_out[1]), float(e_out[0]))
        dth = _norm_bearing_delta(th_in, th_out)
        ds = 0.5 * (li + lo)
        kappa[i] = dth / ds
    return kappa


def _advance_sample_vertex(xy: np.ndarray, i0: int, dist: float) -> int:
    """Walk forward from vertex i0 by dist (m); return vertex index used for curvature."""
    n = len(xy)
    remaining = float(dist)
    i = i0 % n
    for _ in range(n + 2):
        j = (i + 1) % n
        seg = float(np.linalg.norm(xy[j] - xy[i]))
        if seg < MIN_SEG_LEN:
            i = j
            continue
        if remaining <= seg:
            return j
        remaining -= seg
        i = j
    return i0 % n


def _find_xy_columns(fieldnames: list[str]) -> tuple[str, str]:
    lower = {name.lower(): name for name in fieldnames}
    candidates = (
        ("x", "y"),
        ("x_m", "y_m"),
        ("east", "north"),
        ("lon", "lat"),
    )
    for a, b in candidates:
        if a in lower and b in lower:
            return lower[a], lower[b]
    if len(fieldnames) >= 2:
        return fieldnames[0], fieldnames[1]
    raise ValueError("CSV needs recognizable x/y columns or at least two columns.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Append throttle_pwm from lookahead curvature.")
    parser.add_argument("input_csv", type=Path)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output path (default: <input>_throttle.csv)",
    )
    args = parser.parse_args()
    out_path = args.output or args.input_csv.with_name(
        f"{args.input_csv.stem}_throttle{args.input_csv.suffix}"
    )

    with args.input_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("Missing header row.")
        fieldnames = list(reader.fieldnames)
        col_x, col_y = _find_xy_columns(fieldnames)
        rows = list(reader)

    xy = np.array([[float(r[col_x]), float(r[col_y])] for r in rows], dtype=float)
    kappa = vertex_curvatures(xy)

    sampled = np.array(
        [kappa[_advance_sample_vertex(xy, i, LOOKAHEAD)] for i in range(len(xy))],
        dtype=float,
    )
    k_lo = float(np.quantile(sampled, STRAIGHT_KAPPA_QUANTILE))
    k_hi = float(np.quantile(sampled, CURVE_KAPPA_QUANTILE))
    if k_hi <= k_lo:
        k_hi = k_lo + 1e-9

    span = THROTTLE_STRAIGHT - THROTTLE_MAX_CURVE
    pwm_cols: list[int] = []
    for k in sampled:
        u = (float(k) - k_lo) / (k_hi - k_lo)
        u = min(1.0, max(0.0, u))
        t = u ** CURVE_SHARPNESS
        pwm = int(round(THROTTLE_STRAIGHT - t * span))
        pwm_cols.append(pwm)

    have_throttle_col = any(k.lower() == "throttle_pwm" for k in fieldnames)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for row, pwm in zip(rows, pwm_cols):
            out_row: list[object] = []
            for k in fieldnames:
                if k.lower() == "throttle_pwm":
                    out_row.append(pwm)
                else:
                    out_row.append(row.get(k, ""))
            if not have_throttle_col:
                out_row.append(pwm)
            writer.writerow(out_row)

    print(f"Wrote {out_path} ({len(rows)} data rows, no header).")


if __name__ == "__main__":
    main()
