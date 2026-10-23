"""
Truth model + simulated sensors for testing ekf_localizer.

Nothing in here imports anything test-specific; test_ekf.py drives it.

Assumes ekf_localizer_new.py exposes:
    ekf_localizer_new(start_x, start_y, start_heading, dist_between_gps)
    .update(imu_accel, gyro_heading_vel, gps_zed_position,
            gps_gem_position, drive_encoder_vel, cur_time=None)
    .x                 (5,)  [px, py, heading, fwd_vel, gyro_bias]
    .est_error_cov     (5,5)
    module constants ZED_GPS_OFFSET_M, GEM_GPS_OFFSET_M
"""

import math
import copy
import inspect
from dataclasses import dataclass, field

import numpy as np

import ekf_localizer_new
from ekf_localizer_new import EKF_Localizer, wrap


# ----------------------------------------------------------------------------
# config
# ----------------------------------------------------------------------------

@dataclass
class SimConfig:
    duration: float = 60.0

    truth_hz: float = 1000.0     # truth integrator, much faster than the filter
    imu_hz: float = 50.0
    zed_hz: float = 10.0
    gem_hz: float = 5.0
    enc_hz: float = 20.0
    # NOTE: the filter no longer takes a heading measurement - a heading derived
    # from the two antennas is a function of the same two position fixes that
    # get fused directly, so feeding both counted that noise twice. Sensors
    # .gps_heading is kept below for the moving-baseline case, see its comment.
    heading_hz: float = 5.0

    # phase offsets, so the two GPS streams are never synchronised.
    # this is the async situation on the real kart.
    zed_phase: float = 0.0
    gem_phase: float = 0.037

    # geometry
    dist_between_gps: float = 1.0

    # sensor noise (1 sigma), PER SAMPLE - this is what gets added to a reading.
    # accel and gyro default to None and are derived in __post_init__ from the
    # filter's own spec numbers, so the sim and make_Q always describe the same
    # sensor. override with a float to deliberately mismatch them.
    sigma_accel: float = None        # m/s^2
    sigma_gyro: float = None         # rad/s
    sigma_zed: float = 0.00849       # m, per axis, matches R_gps_zed
    sigma_gem: float = 0.01698       # m, per axis, matches R_gps_gem
    sigma_enc: float = 0.10          # m/s, matches R_enc = 0.01

    # the thing the filter is supposed to discover
    true_gyro_bias: float = 0.01     # rad/s

    # initial state error (1 sigma) used to perturb the filter off truth.
    # should match whatever the filter puts on the diagonal of its initial P.
    init_sigma_pos: float = 0.5
    init_sigma_heading: float = math.radians(5.0)
    init_sigma_vel: float = 0.5
    init_sigma_bias: float = math.radians(0.5)

    # sensor dropout windows, list of (sensor_name, t_start, t_end)
    dropouts: list = field(default_factory=list)

    seed: int = 0

    def __post_init__(self):
        accel_density, gyro_density = _filter_noise_densities()
        if self.sigma_accel is None:
            self.sigma_accel = accel_density * math.sqrt(self.imu_hz)
        if self.sigma_gyro is None:
            self.sigma_gyro = gyro_density * math.sqrt(self.imu_hz)

    def lever_arms(self):
        """body-frame offsets if the state tracks the midpoint of the two GPSs,
        baseline along the kart's forward axis."""
        d = self.dist_between_gps
        return (-d / 2.0, 0.0), (d / 2.0, 0.0)


def _filter_noise_densities():
    """the filter's accel/gyro noise numbers, as spec-sheet DENSITIES.

    make_Q uses sigma^2 * dt, which is the right form for a density in
    units/sqrt(Hz). The sim adds noise PER SAMPLE, so the two differ by
    sqrt(imu_hz) - a factor of ~7 at 50 Hz. Getting that backwards does not
    look like a unit error from the outside: it reads as a containment failure
    on fwd_vel and nothing else, so derive the sim's sigmas from the filter's
    numbers rather than writing them down twice.
    """
    probe = EKF_Localizer(0.0, 0.0, 0.0, 1.0)
    return probe.accel_stdev, probe.gyro_stdev


def apply_lever_arms_to_module(cfg):
    """the filter reads its offsets from module-level constants, so set them
    here rather than inside the filter. with (0,0) the lever-arm code path is
    never exercised."""
    zed, gem = cfg.lever_arms()
    ekf_localizer_new.ZED_GPS_OFFSET_M = zed
    ekf_localizer_new.GEM_GPS_OFFSET_M = gem
    return zed, gem


# ----------------------------------------------------------------------------
# control profiles: t -> (forward accel m/s^2, yaw rate rad/s)
# ----------------------------------------------------------------------------

def ctrl_stationary(t):
    return 0.0, 0.0


def ctrl_straight(t):
    return (0.5 if t < 10.0 else 0.0), 0.0


def ctrl_constant_turn(t):
    if t < 5.0:
        return 1.0, 0.0
    return 0.0, 0.3


def ctrl_figure_eight(t):
    """sweeps every heading, both turn directions. the single most useful
    trajectory for catching wrap bugs and heading-dependent jacobian errors."""
    if t < 5.0:
        return 1.0, 0.0
    period = 40.0
    return 0.0, 0.4 * math.sin(2 * math.pi * (t - 5.0) / period)


def ctrl_near_branch_cut(t):
    """parks the heading near +/-pi so measurements land on the far side of
    the wrap."""
    if t < 3.0:
        return 1.0, 0.0
    if t < 8.0:
        return 0.0, math.pi / 5.0      # drive heading up to ~pi
    return 0.0, 0.05 * math.sin(t)     # jitter around the cut


# ----------------------------------------------------------------------------
# truth
# ----------------------------------------------------------------------------

class Truth:
    """unicycle integrated finely enough that its own discretisation error is
    negligible next to the filter's."""

    def __init__(self, cfg, control, x0=0.0, y0=0.0, heading0=0.0, v0=0.0):
        self.cfg = cfg
        self.dt = 1.0 / cfg.truth_hz
        n = int(cfg.duration * cfg.truth_hz) + 1

        self.t = np.arange(n) * self.dt
        self.state = np.zeros((n, 4))      # x, y, heading, v
        self.ctrl = np.zeros((n, 2))       # accel, yaw rate

        x, y, h, v = x0, y0, heading0, v0
        for k in range(n):
            a, w = control(self.t[k])
            self.state[k] = (x, y, h, v)
            self.ctrl[k] = (a, w)

            x += v * math.cos(h) * self.dt
            y += v * math.sin(h) * self.dt
            h = wrap(h + w * self.dt)
            v += a * self.dt

    def index(self, t):
        return min(int(round(t * self.cfg.truth_hz)), len(self.t) - 1)

    def at(self, t):
        """(x, y, heading, v) at time t"""
        return self.state[self.index(t)]

    def control_at(self, t):
        """(accel, yaw rate) at time t"""
        return self.ctrl[self.index(t)]

    def antenna_at(self, t, lever):
        """world position of an antenna mounted at body-frame `lever`."""
        x, y, h, _ = self.at(t)
        lx, ly = lever
        c, s = math.cos(h), math.sin(h)
        return np.array([x + lx * c - ly * s,
                         y + lx * s + ly * c])


# ----------------------------------------------------------------------------
# sensors
# ----------------------------------------------------------------------------

class Sensors:
    def __init__(self, cfg, truth):
        self.cfg = cfg
        self.truth = truth
        self.rng = np.random.default_rng(cfg.seed)
        self.zed_lever, self.gem_lever = cfg.lever_arms()

    def _down(self, name, t):
        for n, t0, t1 in self.cfg.dropouts:
            if n == name and t0 <= t <= t1:
                return True
        return False

    def imu_accel(self, t):
        """body-frame forward accel. the filter uses component 0 as forward
        accel, so component 1 is the lateral term it discards."""
        if self._down("imu", t):
            return None
        a, w = self.truth.control_at(t)
        _, _, _, v = self.truth.at(t)
        ax = a + self.rng.normal(0, self.cfg.sigma_accel)
        ay = v * w + self.rng.normal(0, self.cfg.sigma_accel)   # centripetal
        return (ax, ay)

    def gyro(self, t):
        if self._down("gyro", t):
            return None
        _, w = self.truth.control_at(t)
        return w + self.cfg.true_gyro_bias + self.rng.normal(0, self.cfg.sigma_gyro)

    def gps(self, t, which):
        name, lever, sigma = {
            "zed": ("zed", self.zed_lever, self.cfg.sigma_zed),
            "gem": ("gem", self.gem_lever, self.cfg.sigma_gem),
        }[which]
        if self._down(name, t):
            return None
        # NOTE: noise is added at the ANTENNA, not the kart centre. simulating
        # at the kart centre would mean the lever arm never gets tested.
        return self.truth.antenna_at(t, lever) + self.rng.normal(0, sigma, 2)

    def gps_heading(self, t):
        """heading from the two antenna positions, time-synced (both sampled at
        the same instant).

        NOTE: nothing consumes this any more - the filter gets heading
        observability from the lever-arm column of H_gps instead, and fusing
        both would double count the receivers' noise. Kept because a real
        moving-baseline RTK heading is NOT this quantity: it comes from
        carrier-phase double differencing, which cancels common-mode error and
        is considerably better than differencing two independently-solved
        positions. If you ever fuse that, it is genuinely new information - but
        then fuse heading plus ONE position, never heading plus both.
        """
        if self._down("heading", t):
            return None
        z = self.truth.antenna_at(t, self.zed_lever) + self.rng.normal(0, self.cfg.sigma_zed, 2)
        g = self.truth.antenna_at(t, self.gem_lever) + self.rng.normal(0, self.cfg.sigma_gem, 2)
        d = g - z
        return math.atan2(d[1], d[0])

    def encoder(self, t):
        if self._down("enc", t):
            return None
        _, _, _, v = self.truth.at(t)
        return v + self.rng.normal(0, self.cfg.sigma_enc)


# ----------------------------------------------------------------------------
# runner
# ----------------------------------------------------------------------------

def _time_kwarg(ekf):
    """name of the keyword the filter takes its sample time on, or None if it
    has no such parameter. accept either spelling: getting this wrong silently
    drops the sim clock and the filter falls back to time.monotonic(), which
    makes dt the loop's compute time instead of the sim timestep."""
    params = inspect.signature(ekf.update).parameters
    for name in ("cur_time", "now"):
        if name in params:
            return name
    return None


def run(cfg, control, noise_free=False, perturb_init=True, ekf=None):
    """returns a dict of logged arrays.

    noise_free   -- zero all sensor noise and the true bias
    perturb_init -- start the filter off truth, sampled from the init sigmas.
                    a filter initialised exactly at truth hides divergence bugs.
    ekf          -- pass your own instance to override construction
    """
    if noise_free:
        cfg = SimConfig(**{**cfg.__dict__})
        cfg.sigma_accel = cfg.sigma_gyro = 0.0
        cfg.sigma_zed = cfg.sigma_gem = cfg.sigma_enc = 0.0
        cfg.true_gyro_bias = 0.0
        perturb_init = False

    apply_lever_arms_to_module(cfg)

    truth = Truth(cfg, control)
    sensors = Sensors(cfg, truth)
    rng = np.random.default_rng(cfg.seed + 7919)

    x0, y0, h0, v0 = truth.at(0.0)
    if perturb_init:
        sx = x0 + rng.normal(0, cfg.init_sigma_pos)
        sy = y0 + rng.normal(0, cfg.init_sigma_pos)
        sh = wrap(h0 + rng.normal(0, cfg.init_sigma_heading))
    else:
        sx, sy, sh = x0, y0, h0

    if ekf is None:
        ekf = EKF_Localizer(sx, sy, sh, cfg.dist_between_gps)
    time_kw = _time_kwarg(ekf)

    dt_imu = 1.0 / cfg.imu_hz
    n = int(cfg.duration * cfg.imu_hz)

    log = {k: [] for k in
           ("t", "truth", "est", "P_diag", "P_full", "nis_gps", "nis_enc")}

    def due(t, hz, phase):
        """did a tick of this sensor fall inside the last IMU interval"""
        if hz <= 0:
            return False
        k_now = math.floor((t - phase) * hz)
        k_prev = math.floor((t - dt_imu - phase) * hz)
        return k_now > k_prev

    for i in range(n):
        t = i * dt_imu

        accel = sensors.imu_accel(t)
        gyro = sensors.gyro(t)
        zed = sensors.gps(t, "zed") if due(t, cfg.zed_hz, cfg.zed_phase) else None
        gem = sensors.gps(t, "gem") if due(t, cfg.gem_hz, cfg.gem_phase) else None
        enc = sensors.encoder(t) if due(t, cfg.enc_hz, 0.0) else None

        nis = _nis_for_step(ekf, accel, gyro, t, time_kw, zed, gem, enc)

        if time_kw is not None:
            ekf.update(accel, gyro, zed, gem, enc, **{time_kw: t})
        else:
            ekf.update(accel, gyro, zed, gem, enc)

        tx, ty, th, tv = truth.at(t)
        log["t"].append(t)
        log["truth"].append([tx, ty, th, tv, cfg.true_gyro_bias])
        log["est"].append(ekf.x.copy())
        log["P_diag"].append(np.diag(ekf.est_error_cov).copy())
        log["P_full"].append(ekf.est_error_cov.copy())
        for k, v in nis.items():
            if v is not None:
                log[k].append((t, v))

    out: dict = {k: np.asarray(v) for k, v in log.items()}
    out["error"] = out["truth"] - out["est"]
    out["error"][:, 2] = np.array([wrap(e) for e in out["error"][:, 2]])
    out["cfg"] = cfg
    out["truth_obj"] = truth
    out["ekf"] = ekf
    return out


def _nis_for_step(ekf, accel, gyro, t, time_kw, zed, gem, enc):
    """NIS for whatever measurements arrived this step, scored against the prior.

    "The prior" means after THIS step's prediction and before any measurement
    is fused. Getting that wrong is not a small error: scoring against the
    previous step's posterior leaves a whole propagation interval of real
    motion sitting in the innovation while S is evaluated at a P that has
    already been collapsed by the last GPS fix. The inflation then scales with
    speed, so it looks like a mis-tuned R rather than a measurement artifact.

    The filter has no separate predict() - prediction and fusion both live
    inside update(). So run update() on a throwaway copy with the IMU inputs
    and no measurements: that path propagates x and P and touches nothing else. Copying the filter rather than reimplementing euler
    and F P F.T here means this cannot drift out of sync with it.
    """
    if all(z is None for z in (zed, gem, enc)):
        return {"nis_gps": None, "nis_enc": None}

    prior = copy.deepcopy(ekf)
    if time_kw is not None:
        prior.update(accel, gyro, None, None, None, **{time_kw: t})
    else:
        prior.update(accel, gyro, None, None, None)
    return _nis_terms(prior, zed, gem, enc)


def _nis_terms(ekf, zed, gem, enc):
    """normalised innovation squared, per channel. `ekf` must already be at the
    prior - see _nis_for_step, which is what callers want.

    NIS needs no ground truth, so the same function works on real kart data.
    """
    res: dict[str, float | None] = {"nis_gps": None, "nis_enc": None}

    def nis(innov, H, R):
        S = H @ ekf.est_error_cov @ H.T + R
        return float(innov @ np.linalg.inv(S) @ innov)

    gps_vals = []
    for z, offset, R in ((zed, ekf_localizer_new.ZED_GPS_OFFSET_M, ekf.R_gps_zed),
                         (gem, ekf_localizer_new.GEM_GPS_OFFSET_M, ekf.R_gps_gem)):
        if z is None:
            continue
        pred, H = ekf.make_h_and_H_gps(offset)
        gps_vals.append(nis(np.asarray(z, float) - pred, H, R))
    if gps_vals:
        res["nis_gps"] = float(np.mean(gps_vals))

    if enc is not None:
        innov = np.array([enc - ekf.x[3]])
        res["nis_enc"] = nis(innov, ekf.H_enc, ekf.R_enc)

    return res


# ----------------------------------------------------------------------------
# monte carlo
# ----------------------------------------------------------------------------

def monte_carlo(cfg, control, trials=50):
    """NEES averaged over trials. this is the number that says whether P is
    honest: it should sit near NUM_STATES (5)."""
    t = np.empty(0)
    nees = None
    for k in range(trials):
        c = SimConfig(**{**cfg.__dict__})
        c.seed = cfg.seed + k
        r = run(c, control)
        e = r["error"]
        vals = np.empty(len(e))
        for i in range(len(e)):
            P = r["P_full"][i]
            try:
                vals[i] = e[i] @ np.linalg.solve(P, e[i])
            except np.linalg.LinAlgError:
                vals[i] = np.nan
        nees = vals if nees is None else nees + vals
        if k == 0:
            t = r["t"]
    return t, nees / trials