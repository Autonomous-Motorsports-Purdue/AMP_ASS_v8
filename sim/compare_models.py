"""
Compare vehicle models on the train/test dataset: fit each model on the train runs, score it on the test runs.

Same test as model_check.py, over many runs: from each RTK FIXED GPS fix (moving, no GPS gap), start the
model at the logged position + GPS heading, feed it the logged steering command and speed, and compare the
predicted position 0.5 / 1 / 2 s later against GPS. All windows roll out as one numpy batch.

GPS only (no IMU). Usage (from AMP_ASS_v8/):
    python sim/compare_models.py ~/Downloads/dataset.csv

Outputs (in --out-dir):
    compare_summary.csv   test error per model and horizon (RMS / median / p95 / n)
    compare_speed.csv     test RMS at 1 s per model and speed bin
    compare_params.csv    fitted parameters and rollout cost per model
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from parts.controller import BicycleModel, LagBicycleModel, UndersteerBicycleModel, DynamicBicycleModel
from model_check import HORIZONS_S, MIN_SPEED, MAX_GPS_GAP

DT = 0.02                  # s, rollout step (the logged control loop is ~48 Hz)
RUN_GAP = 1e5              # s between runs on one concatenated timeline, so np.interp never mixes runs
FIT_HORIZON = 1.0          # s, fit to minimise RMS error at this horizon (as model_check does)
SPEED_BINS = [0.5, 2.0, 3.0, 11.0]


# ================= Models =================
# name -> (build(params), extra initial params appended to the refit bicycle's fitted [slope, offset, delay])
def _base(p):
    return dict(kappa_slope=p[0], steering_offset=p[1], delay=abs(p[2]))

MODELS = {
    "straight_line":   (lambda p: BicycleModel(kappa_slope=0.0, delay=0.0), None),
    "textbook_tan":    (lambda p: BicycleModel(mapping="tan", delay=0.0), None),
    "bicycle_current": (lambda p: BicycleModel(), None),  # defaults committed in d7bbff5 (one run)
    "bicycle_refit":   (lambda p: BicycleModel(**_base(p)), []),
    "lag":             (lambda p: LagBicycleModel(tau=abs(p[3]), **_base(p)), [0.15]),
    "understeer":      (lambda p: UndersteerBicycleModel(k_us=abs(p[3]), **_base(p)), [0.02]),
    # cf, cr scaled by 1e4 so Nelder-Mead steps are comparable across parameters
    "dynamic":         (lambda p: DynamicBicycleModel(cf=abs(p[3]) * 1e4, cr=abs(p[4]) * 1e4, **_base(p)), [0.8, 0.8]),
}


# ================= Data =================
def load(path):
    df = pd.read_csv(path, usecols=["run", "t", "steer_cmd", "is_gps_fix", "x_m", "y_m", "yaw_rad", "v_mps", "split"],
                     low_memory=False).sort_values(["run", "t"]).reset_index(drop=True)
    df["T"] = df.groupby("run").ngroup() * RUN_GAP + df["t"]
    df["run_start"] = df.groupby("run")["T"].transform("min")
    fix = df[df["is_gps_fix"] & df["x_m"].notna()].reset_index(drop=True)
    return df, fix


class Windows:
    """Every valid start fix of one split, with its input grid (steps x windows) and GPS truth per horizon."""
    def __init__(self, df, fix, split, every=1):
        t_all = fix["T"].to_numpy()
        h_max = max(HORIZONS_S)
        keep = []
        for k in np.flatnonzero(fix["split"].to_numpy() == split):
            t0 = t_all[k]
            if (fix["v_mps"][k] < MIN_SPEED or np.isnan(fix["yaw_rad"][k])
                    or t0 - fix["run_start"][k] < 1.0):  # room before t0 for the command delay
                continue
            j = np.searchsorted(t_all, t0 + h_max)
            if j >= len(t_all) or np.max(np.diff(t_all[k:j + 1])) > MAX_GPS_GAP:
                continue
            keep.append(k)
        keep = np.array(keep[::every])
        self.run = fix["run"].to_numpy()[keep]
        self.speed = fix["v_mps"].to_numpy()[keep]
        self.x0, self.y0, self.yaw0 = (fix[c].to_numpy()[keep] for c in ("x_m", "y_m", "yaw_rad"))
        t0 = t_all[keep]
        self.grid = t0[None, :] + DT * np.arange(int(round(h_max / DT)))[:, None]
        self.v = np.interp(self.grid, df["T"], df["v_mps"])
        self.cmd_t, self.cmd = df["T"].to_numpy(), df["steer_cmd"].to_numpy()
        self.truth = {h: (np.interp(t0 + h, t_all, fix["x_m"]), np.interp(t0 + h, t_all, fix["y_m"]))
                      for h in HORIZONS_S}

    def errors(self, model):
        """Position error (m) per horizon, each an array over windows."""
        cmd = np.interp(self.grid - model.delay, self.cmd_t, self.cmd)
        traj = model.rollout(self.x0, self.y0, self.yaw0, self.v, cmd, np.full(len(self.grid), DT))
        out = {}
        for h in HORIZONS_S:
            i = int(round(h / DT))
            out[h] = np.hypot(traj[i, 0] - self.truth[h][0], traj[i, 1] - self.truth[h][1])
        return out


# ================= Fit / score =================
def rms(e):
    return float(np.sqrt(np.mean(e ** 2)))


def fit(build, p0, wins):
    cost = lambda p: rms(wins.errors(build(p))[FIT_HORIZON])
    res = minimize(cost, x0=p0, method="Nelder-Mead",
                   options={"xatol": 1e-4, "fatol": 1e-5, "maxiter": 400 * len(p0)})
    return res.x


def run_bootstrap_ci(err, run_ids, n=1000, seed=0):
    """95% CI of RMS error, resampling whole runs (windows inside one run are correlated)."""
    runs = np.unique(run_ids)
    sse = np.array([np.sum(err[run_ids == r] ** 2) for r in runs])
    cnt = np.array([np.sum(run_ids == r) for r in runs])
    idx = np.random.default_rng(seed).integers(0, len(runs), (n, len(runs)))
    boot = np.sqrt(sse[idx].sum(1) / cnt[idx].sum(1))
    return np.percentile(boot, [2.5, 97.5])


def mpc_call_us(model, steps=20):
    """Wall time of one scalar rollout the size an MPC cost function would make."""
    args = (0.0, 0.0, 0.0, np.full(steps, 2.0), np.full(steps, 0.2), np.full(steps, 0.1))
    n = 200
    t = time.perf_counter()
    for _ in range(n):
        model.rollout(*args)
    return (time.perf_counter() - t) / n * 1e6


def sanity_checks():
    v, cmd, dt = np.full(100, 2.0), np.full(100, 0.4), np.full(100, DT)
    base = BicycleModel().rollout(0.0, 0.0, 0.3, v, cmd, dt)
    batch = BicycleModel().rollout(np.zeros(2), np.zeros(2), np.full(2, 0.3),
                                   np.tile(v[:, None], 2), np.tile(cmd[:, None], 2), dt)
    assert np.allclose(batch[:, :, 1], base), "batch rollout differs from scalar rollout"
    assert np.allclose(UndersteerBicycleModel(k_us=0.0).rollout(0.0, 0.0, 0.3, v, cmd, dt), base)
    assert np.allclose(LagBicycleModel(tau=0.2).rollout(0.0, 0.0, 0.3, v, cmd, dt), base), "lag not settled at start"
    dyn = DynamicBicycleModel(cf=1e6, cr=1e6).rollout(0.0, 0.0, 0.3, v, cmd, dt)
    assert np.max(np.hypot(*(dyn[:, :2] - base[:, :2]).T)) < 0.05, "stiff neutral-steer dynamic != kinematic"


# ================= Main =================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_csv")
    parser.add_argument("--out-dir", default="sim/results")
    parser.add_argument("--fit-every", type=int, default=3, help="use every Nth train window for fitting")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    sanity_checks()

    df, fix = load(args.dataset_csv)
    train = Windows(df, fix, "train", every=args.fit_every)
    test = Windows(df, fix, "test")
    print(f"{len(df)} rows, {len(fix)} GPS fixes -> {len(train.x0)} train windows (fit), "
          f"{len(test.x0)} test windows across {len(np.unique(test.run))} test runs")

    refit_p = fit(MODELS["bicycle_refit"][0], [-0.542, 0.063, 0.11], train)
    summary, speed, params = [], [], []
    for name, (build, extra) in MODELS.items():
        t = time.perf_counter()
        p = [] if extra is None else (refit_p if extra == [] else fit(build, list(refit_p) + extra, train))
        model = build(p)
        err = test.errors(model)
        for h in HORIZONS_S:
            lo, hi = run_bootstrap_ci(err[h], test.run)
            summary.append({"model": name, "horizon_s": h, "rms": rms(err[h]), "rms_ci_lo": lo, "rms_ci_hi": hi,
                            "median": np.median(err[h]), "p95": np.percentile(err[h], 95), "n": len(err[h])})
        bins = np.digitize(test.speed, SPEED_BINS[1:-1])
        for b in range(len(SPEED_BINS) - 1):
            e = err[FIT_HORIZON][bins == b]
            speed.append({"model": name, "speed_bin": f"{SPEED_BINS[b]}-{SPEED_BINS[b + 1]} m/s",
                          "rms": rms(e) if len(e) else np.nan, "n": len(e)})
        params.append({"model": name, "n_fitted": len(p), "mpc_call_us": mpc_call_us(model),
                       **{k: v for k, v in vars(model).items() if isinstance(v, (int, float))}})
        print(f"{name:16s} 1 s RMS {rms(err[FIT_HORIZON]):.3f} m  ({time.perf_counter() - t:.1f} s)")

    summary, speed, params = pd.DataFrame(summary), pd.DataFrame(speed), pd.DataFrame(params)
    summary.to_csv(os.path.join(args.out_dir, "compare_summary.csv"), index=False)
    speed.to_csv(os.path.join(args.out_dir, "compare_speed.csv"), index=False)
    params.to_csv(os.path.join(args.out_dir, "compare_params.csv"), index=False)
    print("\nTest RMS position error (m):")
    print(summary.pivot(index="model", columns="horizon_s", values="rms").reindex(MODELS).round(3).to_string())
    print("\nTest RMS at 1 s by speed (m):")
    print(speed.pivot(index="model", columns="speed_bin", values="rms").reindex(MODELS).round(3).to_string())
    print("\nParameters:")
    print(params.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
