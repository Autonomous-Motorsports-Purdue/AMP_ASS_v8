"""
Checks on EK_Localizer. Run directly:

    python test_ekf.py              # all checks, text only
    python test_ekf.py --plot       # also draw the consistency plots
    python test_ekf.py jacobians    # single check by name

Order matters: run the jacobian and noise-free checks first. Those catch logic
bugs. The consistency checks only tell you about tuning, and tuning numbers are
meaningless if the maths is wrong.
"""

import math
import sys

import numpy as np

import ekf_localizer_new
from ekf_localizer_new import EKF_Localizer, NUM_STATES, wrap
import sim
from sim import (SimConfig, run, monte_carlo,
                 ctrl_stationary, ctrl_straight, ctrl_constant_turn,
                 ctrl_figure_eight, ctrl_near_branch_cut)


STATE_NAMES = ["pos_x (m)", "pos_y (m)", "heading (rad)",
               "fwd_vel (m/s)", "gyro_bias (rad/s)"]

_results = []


def report(name, ok, detail=""):
    _results.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))


# ----------------------------------------------------------------------------
# 1. jacobians
# ----------------------------------------------------------------------------

def check_jacobians(eps=1e-6, tol=1e-5):
    """finite-difference F and H against the functions they linearise.

    highest value per minute of anything here: catches sign errors and dropped
    terms outright, with no tuning involved.
    """
    rng = np.random.default_rng(3)
    cfg = SimConfig()
    sim.apply_lever_arms_to_module(cfg)
    ekf = EKF_Localizer(0.0, 0.0, 0.0, cfg.dist_between_gps)

    worst_F = 0.0
    worst_H = 0.0

    for _ in range(25):
        x = np.array([rng.uniform(-50, 50), rng.uniform(-50, 50),
                      rng.uniform(-math.pi, math.pi),
                      rng.uniform(-10, 10), rng.uniform(-0.05, 0.05)])
        dt = rng.uniform(0.005, 0.05)
        accel = rng.uniform(-2, 2)
        gyro = rng.uniform(-1, 1)

        # --- F ---
        F = ekf.build_F(x[2], x[3], dt)
        f0 = ekf.predict_next_state(x, dt, accel, gyro)
        for j in range(NUM_STATES):
            xp = x.copy()
            xp[j] += eps
            f1 = ekf.predict_next_state(xp, dt, accel, gyro)
            num = (f1 - f0) / eps
            num[2] = wrap(f1[2] - f0[2]) / eps     # heading row wraps
            worst_F = max(worst_F, np.max(np.abs(num - F[:, j])))

        # --- H ---
        for offset in (ekf_localizer_new.ZED_GPS_OFFSET_M, ekf_localizer_new.GEM_GPS_OFFSET_M):
            ekf.x = x.copy()
            h0, H = ekf.make_h_and_H_gps(offset)
            for j in range(NUM_STATES):
                ekf.x = x.copy()
                ekf.x[j] += eps
                h1, _ = ekf.make_h_and_H_gps(offset)
                num = (np.asarray(h1) - np.asarray(h0)) / eps
                worst_H = max(worst_H, np.max(np.abs(num - H[:, j])))

    report("jacobian F matches predict_next_state", worst_F < tol, f"max err {worst_F:.2e}")
    report("jacobian H matches make_h_and_H_gps", worst_H < tol, f"max err {worst_H:.2e}")


# ----------------------------------------------------------------------------
# 2. structural invariants
# ----------------------------------------------------------------------------

def check_invariants():
    """P must stay symmetric and positive definite for the whole run."""
    cfg = SimConfig(duration=40.0)
    r = run(cfg, ctrl_figure_eight)

    worst_asym = 0.0
    chol_fail = 0
    for P in r["P_full"]:
        worst_asym = max(worst_asym, np.max(np.abs(P - P.T)))
        try:
            np.linalg.cholesky(P)
        except np.linalg.LinAlgError:
            chol_fail += 1

    report("P stays symmetric", worst_asym < 1e-10, f"max |P - P.T| = {worst_asym:.2e}")
    report("P stays positive definite", chol_fail == 0,
           f"{chol_fail}/{len(r['P_full'])} steps failed cholesky")


# ----------------------------------------------------------------------------
# 3. noise-free
# ----------------------------------------------------------------------------

def check_noise_free(tol_pos=0.05, tol_head=math.radians(1.0)):
    """zero noise, perfect init. any drift is a logic bug, not tuning.

    tolerance is not machine precision because the filter euler-integrates at
    50 Hz while truth runs at 1 kHz. what you are looking for is a flat error,
    not a growing one.
    """
    cfg = SimConfig(duration=40.0)
    r = run(cfg, ctrl_figure_eight, noise_free=True)
    e = r["error"]

    pos_err = np.hypot(e[:, 0], e[:, 1])
    head_err = np.abs(e[:, 2])

    report("noise-free position tracks truth", pos_err.max() < tol_pos,
           f"max {pos_err.max():.4f} m")
    report("noise-free heading tracks truth", head_err.max() < tol_head,
           f"max {math.degrees(head_err.max()):.3f} deg")


# ----------------------------------------------------------------------------
# 4. bias observability
# ----------------------------------------------------------------------------

def check_bias_convergence():
    """the direct test of whether the P[2,4] coupling is alive.

    nothing measures bias, so it can only be corrected through its correlation
    with heading. if the estimate sits at 0 forever, Q[4,4] is the culprit.
    """
    cfg = SimConfig(duration=90.0, true_gyro_bias=0.01)
    r = run(cfg, ctrl_figure_eight)

    est = r["est"][:, 4]
    final = np.mean(est[-int(len(est) * 0.1):])
    moved = abs(final) > 0.1 * abs(cfg.true_gyro_bias)
    close = abs(final - cfg.true_gyro_bias) < 0.3 * abs(cfg.true_gyro_bias)

    report("gyro bias estimate moves off zero", moved,
           f"final {final:.5f} vs true {cfg.true_gyro_bias:.5f}")
    report("gyro bias converges to truth", close,
           f"final {final:.5f} vs true {cfg.true_gyro_bias:.5f}")


# ----------------------------------------------------------------------------
# 5. consistency
# ----------------------------------------------------------------------------

def check_containment(target=0.95, slack=0.10):
    """fraction of error inside +/-2 sigma. want about 95%.

    below  -> Q too small, filter overconfident
    above  -> Q too large, filter throwing away information
    """
    cfg = SimConfig(duration=60.0)
    r = run(cfg, ctrl_figure_eight)

    # skip the transient while the filter pulls in from its perturbed start
    start = int(0.2 * len(r["t"]))
    e = r["error"][start:]
    sd = np.sqrt(r["P_diag"][start:])

    ok_all = True
    for i, nm in enumerate(STATE_NAMES):
        frac = np.mean(np.abs(e[:, i]) <= 2.0 * sd[:, i])
        good = abs(frac - target) <= slack
        ok_all &= good
        print(f"       {nm:<20s} {frac*100:5.1f}% inside 2-sigma "
              f"{'' if good else '  <-- tune Q'}")
    report("2-sigma containment near 95% on all states", ok_all)


def check_nees(trials=30, tol=1.5):
    """NEES = e.T P^-1 e, averaged over trials. should sit near NUM_STATES.

    the single number that says whether P is honest.
    """
    cfg = SimConfig(duration=40.0)
    t, nees = monte_carlo(cfg, ctrl_figure_eight, trials=trials)
    start = int(0.2 * len(t))
    mean = np.nanmean(nees[start:])
    report(f"NEES near {NUM_STATES} ({trials} trials)", abs(mean - NUM_STATES) < tol,
           f"mean {mean:.2f}")


def check_nis(tol=0.6):
    """NIS per channel, should average to that channel's measurement dim.

    unlike NEES this needs no ground truth, so the same check runs on real
    kart data later.
    """
    cfg = SimConfig(duration=60.0)
    r = run(cfg, ctrl_figure_eight)

    for key, dim in (("nis_gps", 2), ("nis_heading", 1), ("nis_enc", 1)):
        arr = r[key]
        if len(arr) == 0:
            report(f"{key} average near {dim}", False, "no samples")
            continue
        vals = arr[:, 1]
        vals = vals[int(0.2 * len(vals)):]
        mean = float(np.mean(vals))
        report(f"{key} average near {dim}", abs(mean - dim) < tol, f"mean {mean:.2f}")


# ----------------------------------------------------------------------------
# 6. scenarios
# ----------------------------------------------------------------------------

def check_stationary():
    cfg = SimConfig(duration=60.0)
    r = run(cfg, ctrl_stationary)
    e = r["error"][int(0.3 * len(r["t"])):]
    drift = np.hypot(e[:, 0], e[:, 1]).max()
    report("stationary: no position runaway", drift < 1.0, f"max {drift:.3f} m")


def check_zero_offsets():
    """with zero lever arms H[:,2] is zero and this should reduce exactly to
    the pre-lever-arm filter."""
    cfg = SimConfig(duration=30.0, dist_between_gps=1.0)
    saved = (ekf_localizer_new.ZED_GPS_OFFSET_M, ekf_localizer_new.GEM_GPS_OFFSET_M)
    try:
        r = run(cfg, ctrl_straight)
        sim.apply_lever_arms_to_module(cfg)
        ekf_localizer_new.ZED_GPS_OFFSET_M = (0.0, 0.0)
        ekf_localizer_new.GEM_GPS_OFFSET_M = (0.0, 0.0)
        ekf = EKF_Localizer(0.0, 0.0, 0.0, cfg.dist_between_gps)
        _, H = ekf.make_h_and_H_gps((0.0, 0.0))
        report("zero offsets give zero heading column in H",
               np.allclose(H[:, 2], 0.0))
    finally:
        ekf_localizer_new.ZED_GPS_OFFSET_M, ekf_localizer_new.GEM_GPS_OFFSET_M = saved


def check_branch_cut():
    """heading parked near +/-pi, so measurements land on the far side of the
    wrap. a missing wrap shows up as a huge heading error here and nowhere
    else."""
    cfg = SimConfig(duration=40.0)
    r = run(cfg, ctrl_near_branch_cut)
    start = int(0.4 * len(r["t"]))
    worst = np.abs(r["error"][start:, 2]).max()
    report("heading survives the +/-pi branch cut", worst < math.radians(15.0),
           f"max {math.degrees(worst):.2f} deg")


def check_gps_dropout():
    """kill both receivers for 10 s. P should grow while dead reckoning and the
    estimate should snap back on reacquisition without overshooting."""
    cfg = SimConfig(duration=60.0,
                    dropouts=[("zed", 20.0, 30.0), ("gem", 20.0, 30.0),
                              ("heading", 20.0, 30.0)])
    r = run(cfg, ctrl_straight)
    t = r["t"]
    pvar = r["P_diag"][:, 0]

    i_before = np.searchsorted(t, 19.0)
    i_during = np.searchsorted(t, 29.5)
    i_after = np.searchsorted(t, 35.0)

    grew = pvar[i_during] > pvar[i_before] * 2.0
    shrank = pvar[i_after] < pvar[i_during]
    recovered = np.hypot(*r["error"][i_after, :2]) < 1.0

    report("dropout: P grows while dead reckoning", grew,
           f"{pvar[i_before]:.4f} -> {pvar[i_during]:.4f}")
    report("dropout: P recovers after reacquisition", shrank,
           f"{pvar[i_during]:.4f} -> {pvar[i_after]:.4f}")
    report("dropout: estimate recovers after reacquisition", recovered,
           f"err {np.hypot(*r['error'][i_after, :2]):.3f} m")


# ----------------------------------------------------------------------------
# plots
# ----------------------------------------------------------------------------

def plot_all():
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plots")
        return

    cfg = SimConfig(duration=60.0)
    r = run(cfg, ctrl_figure_eight)
    t, e, sd = r["t"], r["error"], np.sqrt(r["P_diag"])

    fig, axes = plt.subplots(3, 2, figsize=(13, 9))

    for i, nm in enumerate(STATE_NAMES):
        ax = axes.flat[i]
        ax.plot(t, e[:, i], lw=0.8, label="error")
        ax.plot(t, 2 * sd[:, i], "r--", lw=0.8, label="+/-2 sigma")
        ax.plot(t, -2 * sd[:, i], "r--", lw=0.8)
        ax.set_title(nm)
        ax.set_xlabel("t (s)")
        if i == 0:
            ax.legend(fontsize=8)

    ax = axes.flat[5]
    ax.plot(r["truth"][:, 0], r["truth"][:, 1], lw=1.0, label="truth")
    ax.plot(r["est"][:, 0], r["est"][:, 1], lw=0.8, label="estimate")
    ax.set_title("trajectory")
    ax.axis("equal")
    ax.legend(fontsize=8)

    fig.tight_layout()

    t2, nees = monte_carlo(SimConfig(duration=40.0), ctrl_figure_eight, trials=30)
    fig2, ax2 = plt.subplots(figsize=(9, 3.2))
    ax2.plot(t2, nees, lw=0.8)
    ax2.axhline(NUM_STATES, color="r", ls="--", lw=0.8, label=f"ideal = {NUM_STATES}")
    ax2.set_title("NEES (30-trial average)")
    ax2.set_xlabel("t (s)")
    ax2.legend(fontsize=8)
    fig2.tight_layout()

    plt.show()


# ----------------------------------------------------------------------------

CHECKS = {
    "jacobians": check_jacobians,
    "invariants": check_invariants,
    "noise_free": check_noise_free,
    "bias": check_bias_convergence,
    "containment": check_containment,
    "nees": check_nees,
    "nis": check_nis,
    "stationary": check_stationary,
    "zero_offsets": check_zero_offsets,
    "branch_cut": check_branch_cut,
    "dropout": check_gps_dropout,
}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    names = args if args else list(CHECKS)

    for nm in names:
        if nm not in CHECKS:
            print(f"unknown check: {nm}  (have: {', '.join(CHECKS)})")
            return 2
        CHECKS[nm]()

    failed = [n for n, ok in _results if not ok]
    print(f"\n{len(_results) - len(failed)}/{len(_results)} passed")
    if failed:
        print("failed: " + ", ".join(failed))

    if "--plot" in sys.argv:
        plot_all()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())