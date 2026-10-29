"""
Bicycle model check: does the kinematic bicycle model predict how the real car moves?

Replays a logged run: from each RTK FIXED GPS sample, start the model at the car's real
position + GPS heading, feed it the logged steering command and speed, and compare
the predicted position 0.5 / 1 / 2 s later against where GPS says the car actually was.

GPS only (no IMU). Usage (from AMP_ASS_v8/):
    python sim/model_check.py ~/Downloads/information/2026-05-19-10-58-55.csv

Outputs (in --out-dir):
    model_check_predictions.csv  one row per (model, start sample, horizon): predicted vs actual
    model_check_summary.csv      RMS / median / max error per model and horizon
    model_check.png              predictions on the driven path + error vs horizon
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from parts.controller import BicycleModel

R_EARTH = 6378137.0
GPS_FIX = "RTK FIXED"
HORIZONS_S = [0.5, 1.0, 2.0]
MIN_SPEED = 0.5            # m/s, skip windows where the car is (nearly) stopped
MAX_GPS_GAP = 0.5          # s, skip windows with a gap in RTK FIXED coverage


# ================= Load log =================
def load_log(path):
    df = pd.read_csv(path, sep=None, engine="python")
    df["t"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d-%H-%M-%S.%f")
    df["t"] = (df["t"] - df["t"].iloc[0]).dt.total_seconds()

    # steering commands are valid regardless of fix quality: keep every control row
    loop = df[["t", "steering"]].dropna().reset_index(drop=True)

    # positions only from RTK FIXED; GPS updates ~4 Hz vs ~48 Hz control loop, so keep rows where the fix moved
    fixed = df[df["fix"] == GPS_FIX]
    moved = (fixed["latitude"].diff() != 0) | (fixed["longitude"].diff() != 0)
    gps = fixed[moved].reset_index(drop=True)

    lat0, lon0 = gps["latitude"].iloc[0], gps["longitude"].iloc[0]
    gps["x"] = np.radians(gps["longitude"] - lon0) * R_EARTH * np.cos(np.radians(lat0))
    gps["y"] = np.radians(gps["latitude"] - lat0) * R_EARTH
    # gps_heading is a compass bearing (0 = north, clockwise) -> math yaw (0 = east, CCW)
    gps["yaw"] = np.radians(90.0 - gps["gps_heading"])
    # speed from consecutive fixes (gps_speed column is too sparse)
    dist = np.hypot(np.gradient(gps["x"]), np.gradient(gps["y"]))
    gps["v"] = dist / np.gradient(gps["t"])

    loop["v"] = np.interp(loop["t"], gps["t"], gps["v"])
    return gps, loop, len(df)


# ================= Replay =================
def windows(gps, horizons):
    """(start index, horizon) pairs with the car moving and unbroken RTK FIXED coverage."""
    t = gps["t"].to_numpy()
    out = []
    for k in range(len(gps)):
        if gps["v"][k] < MIN_SPEED or np.isnan(gps["yaw"][k]):
            continue
        for h in horizons:
            if t[k] + h > t[-1]:
                continue
            j = np.searchsorted(t, t[k] + h)
            if np.max(np.diff(t[k:j + 1])) > MAX_GPS_GAP:
                continue
            out.append((k, h))
    return out


def predict(model, gps, loop, k, h):
    """Roll the model forward h seconds from GPS sample k using the logged commands."""
    t0 = gps["t"][k]
    sel = (loop["t"] >= t0) & (loop["t"] < t0 + h)
    ts = loop["t"][sel].to_numpy()
    cmd = np.interp(ts - model.delay, loop["t"], loop["steering"])
    dts = np.diff(np.append(ts, t0 + h))
    return model.rollout(gps["x"][k], gps["y"][k], gps["yaw"][k], loop["v"][sel], cmd, dts)


def evaluate(model, gps, loop, wins):
    rows = []
    for k, h in wins:
        traj = predict(model, gps, loop, k, h)
        t_end = gps["t"][k] + h
        x_true = np.interp(t_end, gps["t"], gps["x"])
        y_true = np.interp(t_end, gps["t"], gps["y"])
        rows.append({
            "start_t": gps["t"][k], "horizon_s": h,
            "start_x": gps["x"][k], "start_y": gps["y"][k], "speed_mps": gps["v"][k],
            "pred_x": traj[-1, 0], "pred_y": traj[-1, 1],
            "actual_x": x_true, "actual_y": y_true,
            "error_m": np.hypot(traj[-1, 0] - x_true, traj[-1, 1] - y_true),
        })
    return pd.DataFrame(rows)


def fit_model(gps, loop):
    """Fit curvature slope, steering offset and delay to minimise 1 s prediction error."""
    wins = windows(gps, [1.0])
    def cost(p):
        model = BicycleModel(kappa_slope=p[0], steering_offset=p[1], delay=max(p[2], 0.0))
        e = evaluate(model, gps, loop, wins)["error_m"]
        return np.sqrt(np.mean(e ** 2))
    res = minimize(cost, x0=[-0.223, 0.097, 0.1], method="Nelder-Mead",
                   options={"xatol": 1e-3, "fatol": 1e-4, "maxiter": 200})
    return BicycleModel(kappa_slope=res.x[0], steering_offset=res.x[1], delay=max(res.x[2], 0.0))


# ================= Main =================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_csv")
    parser.add_argument("--out-dir", default="sim/results")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    gps, loop, n_rows = load_log(args.log_csv)
    wins = windows(gps, HORIZONS_S)
    print(f"{n_rows} log rows -> {len(gps)} RTK FIXED GPS fixes, {len(loop)} control rows, "
          f"{len(wins)} test windows")

    fitted = fit_model(gps, loop)
    print(f"fitted: kappa = {fitted.kappa_slope:.3f} * (cmd - {fitted.steering_offset:.3f}), "
          f"delay = {fitted.delay:.2f} s")
    models = {
        "straight_line": BicycleModel(kappa_slope=0.0, delay=0.0),
        "textbook_tan": BicycleModel(mapping="tan", delay=0.0),
        "pp_empirical": BicycleModel(kappa_slope=-0.223, steering_offset=0.097, delay=0.0),
        "controller_default": BicycleModel(),
        "fitted": fitted,
    }

    preds = []
    for name, model in models.items():
        p = evaluate(model, gps, loop, wins)
        p.insert(0, "model", name)
        preds.append(p)
    preds = pd.concat(preds, ignore_index=True)
    summary = (preds.groupby(["model", "horizon_s"])["error_m"]
               .agg(rms=lambda e: np.sqrt(np.mean(e ** 2)), median="median", max="max", n="count")
               .reset_index())

    preds.to_csv(os.path.join(args.out_dir, "model_check_predictions.csv"), index=False)
    summary.to_csv(os.path.join(args.out_dir, "model_check_summary.csv"), index=False)
    print("\nRMS position error (m):")
    print(summary.pivot(index="model", columns="horizon_s", values="rms")
          .reindex(models.keys()).round(3).to_string())

    # ---- plots: example predictions on the driven path + error growth per model ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.plot(gps["x"], gps["y"], "k.", ms=2, label="GPS RTK FIXED (actual)")
    starts = sorted({k for k, h in wins if h == 2.0})[::25]
    for i, k in enumerate(starts):
        traj = predict(fitted, gps, loop, k, 2.0)
        ax1.plot(traj[:, 0], traj[:, 1], "r-", lw=1.5, label="fitted model, 2 s ahead" if i == 0 else None)
    ax1.set_aspect("equal"); ax1.grid(True); ax1.legend()
    ax1.set_xlabel("east (m)"); ax1.set_ylabel("north (m)")
    ax1.set_title("Bicycle predictions vs actual GPS path")

    for name in models:
        s = summary[summary["model"] == name]
        ax2.plot(s["horizon_s"], s["rms"], "o-", label=name)
    ax2.set_xlabel("prediction horizon (s)"); ax2.set_ylabel("RMS position error (m)")
    ax2.set_title("Prediction error vs horizon"); ax2.grid(True); ax2.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "model_check.png"), dpi=120)
    print(f"\nresults written to {args.out_dir}/")


if __name__ == "__main__":
    main()
