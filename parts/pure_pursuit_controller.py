import math

import numpy as np


def is_valid_number(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def normalize_angle_rad(angle_rad):
    return (float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi


def normalize_angle_deg(angle_deg):
    return (float(angle_deg) + 180.0) % 360.0 - 180.0


PP_DEBUG_FIELDS = [
    "reverse_path",
    "resynced_this_cycle",
    "resync_reason",
    "target_behind_counter",
    "resync_dist_m",
    "resync_heading_error_deg",
    "closest_idx",
    "target_idx",
    "closest_dist_m",
    "lookahead_m",
    "yaw_deg",
    "target_x_m",
    "target_y_m",
    "target_heading_deg",
    "heading_error_deg",
    "target_kappa",
    "pp_curvature",
    "x_vehicle",
    "y_vehicle",
    "speed_mps",
    "target_behind",
    "curvature_to_steering_mode",
    "empirical_kappa_slope",
    "empirical_steering_offset",
    "bicycle_steering_cmd",
    "empirical_steering_cmd",
    "selected_steering_cmd",
    "throttle_erpm",
    "steering_norm",
]


class PurePursuitController:
    """
    Minimal GPS Pure Pursuit controller for the current Donkeycar stack.

    Inputs:
        x, y: local ENU position in meters
        yaw_deg: math-frame yaw, degrees, 0=east, +CCW
        speed_mps: current vehicle speed estimate

    Outputs:
        throttle_erpm, steering_norm, pp_debug
    """

    def __init__(
        self,
        path_csv,
        wheelbase_m=1.05,
        steer_max_deg=25.0,
        lookahead_time_s=0.6,
        min_lookahead_m=4.0,
        max_lookahead_m=10.0,
        search_window=80,
        max_resync_dist_m=15.0,
        resync_dist_m=1.5,
        resync_heading_error_deg=120.0,
        target_behind_resync_cycles=3,
        fallback_erpm=1500,
        throttle_floor_erpm=0,
        throttle_ceiling_erpm=4500,
        behind_target_erpm=1200,
        off_path_slowdown_m=1.5,
        off_path_stop_m=3.0,
        off_path_slowdown_erpm=1200,
        reverse_path=False,
        curvature_to_steering="empirical",
        empirical_kappa_slope=-0.223,
        empirical_steering_offset=0.097,
        steering_sign=1.0,
        verbose=False,
    ):
        line = self._load_line(path_csv, fallback_erpm, reverse_path=reverse_path)
        if len(line) < 3:
            raise ValueError(f"PurePursuitController needs at least 3 points: {path_csv}")

        self.s_m = line[:, 0]
        self.path_x_m = line[:, 1]
        self.path_y_m = line[:, 2]
        self.path_psi_rad = line[:, 3]
        self.path_kappa = line[:, 4]
        self.target_erpm = line[:, 5].astype(int)

        self.wheelbase_m = float(wheelbase_m)
        self.steer_max_rad = math.radians(float(steer_max_deg))
        self.lookahead_time_s = float(lookahead_time_s)
        self.min_lookahead_m = float(min_lookahead_m)
        self.max_lookahead_m = float(max_lookahead_m)
        self.search_window = int(search_window)
        self.max_resync_dist_m = float(max_resync_dist_m)
        self.resync_dist_m = float(resync_dist_m)
        self.resync_heading_error_deg = float(resync_heading_error_deg)
        self.target_behind_resync_cycles = max(1, int(target_behind_resync_cycles))
        self.fallback_erpm = int(fallback_erpm)
        self.throttle_floor_erpm = int(throttle_floor_erpm)
        self.throttle_ceiling_erpm = int(throttle_ceiling_erpm)
        self.behind_target_erpm = int(behind_target_erpm)
        self.off_path_slowdown_m = float(off_path_slowdown_m)
        self.off_path_stop_m = float(off_path_stop_m)
        self.off_path_slowdown_erpm = int(off_path_slowdown_erpm)
        self.reverse_path = bool(reverse_path)
        self.curvature_to_steering = str(curvature_to_steering).lower()
        if self.curvature_to_steering not in ("bicycle", "empirical"):
            raise ValueError(
                "curvature_to_steering must be 'bicycle' or 'empirical'"
            )
        self.empirical_kappa_slope = float(empirical_kappa_slope)
        self.empirical_steering_offset = float(empirical_steering_offset)
        self.steering_sign = float(steering_sign)
        self.verbose = bool(verbose)

        self.closest_idx = 0
        self.initial_sync_done = False
        self.target_behind_counter = 0
        self.last_debug = None

    @staticmethod
    def _load_raw_path(path_csv):
        path = np.genfromtxt(
            path_csv,
            delimiter=",",
            dtype=float,
            encoding="utf-8",
            skip_header=0,
        )
        if path.ndim == 1:
            path = np.reshape(path, (1, -1))

        if not np.isfinite(path[0]).all():
            path = np.genfromtxt(
                path_csv,
                delimiter=",",
                dtype=float,
                encoding="utf-8",
                skip_header=1,
            )
            if path.ndim == 1:
                path = np.reshape(path, (1, -1))

        if path.size == 0 or path.shape[1] < 2:
            raise ValueError(f"Could not parse XY path from {path_csv}")
        return np.asarray(path, dtype=float)

    @staticmethod
    def _looks_like_erpm(values):
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return False
        if np.nanmax(np.abs(finite)) >= 50.0:
            return True
        return bool(np.nanmedian(np.abs(finite)) >= 25.0)

    @classmethod
    def _load_line(cls, path_csv, fallback_erpm, reverse_path=False):
        raw = cls._load_raw_path(path_csv)
        if reverse_path:
            raw = raw[::-1].copy()
        xy = np.asarray(raw[:, :2], dtype=float)
        if not np.isfinite(xy).all():
            raise ValueError(f"Non-finite XY values in {path_csv}")

        n = len(xy)
        target_erpm = np.full(n, int(fallback_erpm), dtype=int)

        if raw.shape[1] == 3:
            third = raw[:, 2]
            if cls._looks_like_erpm(third):
                target_erpm = np.rint(third).astype(int)
        elif raw.shape[1] >= 4:
            target_erpm = np.rint(raw[:, -1]).astype(int)

        s_m = cls._build_path_s(xy)
        psi_rad = cls._build_path_heading(xy)
        kappa_radpm = cls._build_path_curvature(xy, psi_rad)

        return np.column_stack(
            [s_m, xy[:, 0], xy[:, 1], psi_rad, kappa_radpm, target_erpm]
        )

    @staticmethod
    def _build_path_s(xy):
        segment_lengths = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))
        return np.concatenate(([0.0], np.cumsum(segment_lengths)))

    @staticmethod
    def _build_path_heading(xy):
        n = len(xy)
        headings = np.zeros(n, dtype=float)
        for i in range(n):
            prev_i = (i - 1) % n
            next_i = (i + 1) % n
            dx = float(xy[next_i, 0] - xy[prev_i, 0])
            dy = float(xy[next_i, 1] - xy[prev_i, 1])
            if math.hypot(dx, dy) < 1e-6:
                dx = float(xy[next_i, 0] - xy[i, 0])
                dy = float(xy[next_i, 1] - xy[i, 1])
            headings[i] = math.atan2(dy, dx)
        return headings

    @staticmethod
    def _build_path_curvature(xy, headings):
        n = len(xy)
        curvature = np.zeros(n, dtype=float)
        for i in range(n):
            prev_i = (i - 1) % n
            next_i = (i + 1) % n
            seg_prev = math.hypot(
                float(xy[i, 0] - xy[prev_i, 0]),
                float(xy[i, 1] - xy[prev_i, 1]),
            )
            seg_next = math.hypot(
                float(xy[next_i, 0] - xy[i, 0]),
                float(xy[next_i, 1] - xy[i, 1]),
            )
            denom = max(seg_prev + seg_next, 1e-6)
            dpsi = normalize_angle_rad(headings[next_i] - headings[prev_i])
            curvature[i] = dpsi / denom
        return curvature

    def _full_resync_idx(self, x_m, y_m):
        distances = np.hypot(self.path_x_m - x_m, self.path_y_m - y_m)
        best_idx = int(np.argmin(distances))
        return best_idx, float(distances[best_idx])

    def _find_closest_idx_forward(self, x_m, y_m):
        if not self.initial_sync_done:
            self.closest_idx, closest_dist_m = self._full_resync_idx(x_m, y_m)
            self.initial_sync_done = True
            return self.closest_idx, closest_dist_m

        current_dist_m = math.hypot(
            float(x_m - self.path_x_m[self.closest_idx]),
            float(y_m - self.path_y_m[self.closest_idx]),
        )
        if current_dist_m > self.max_resync_dist_m:
            self.closest_idx, closest_dist_m = self._full_resync_idx(x_m, y_m)
            return self.closest_idx, closest_dist_m

        n = len(self.path_x_m)
        window = (self.closest_idx + np.arange(self.search_window + 1)) % n
        distances = np.hypot(self.path_x_m[window] - x_m, self.path_y_m[window] - y_m)
        best_local = int(np.argmin(distances))
        self.closest_idx = int(window[best_local])
        return self.closest_idx, float(distances[best_local])

    def _pick_lookahead_target(self, closest_idx, lookahead_m):
        n = len(self.path_x_m)
        remaining = float(lookahead_m)
        i = int(closest_idx)

        for _ in range(n + 2):
            j = (i + 1) % n
            seg_dx = float(self.path_x_m[j] - self.path_x_m[i])
            seg_dy = float(self.path_y_m[j] - self.path_y_m[i])
            seg_len = math.hypot(seg_dx, seg_dy)

            if seg_len < 1e-6:
                i = j
                continue

            if remaining <= seg_len:
                ratio = remaining / seg_len
                target_x_m = float(self.path_x_m[i] + ratio * seg_dx)
                target_y_m = float(self.path_y_m[i] + ratio * seg_dy)
                return j, target_x_m, target_y_m

            remaining -= seg_len
            i = j

        return closest_idx, float(self.path_x_m[closest_idx]), float(self.path_y_m[closest_idx])

    def _compute_tracking_state(
        self, x_m, y_m, yaw_deg, yaw_rad, speed_mps, closest_idx, closest_dist_m
    ):
        lookahead_m = float(
            np.clip(
                speed_mps * self.lookahead_time_s,
                self.min_lookahead_m,
                self.max_lookahead_m,
            )
        )
        target_idx, target_x_m, target_y_m = self._pick_lookahead_target(
            closest_idx, lookahead_m
        )
        _, curvature, x_vehicle, y_vehicle, target_behind = (
            self._pure_pursuit_steering(
                x_m,
                y_m,
                yaw_rad,
                target_x_m,
                target_y_m,
            )
        )
        target_heading_deg = float(np.degrees(self.path_psi_rad[target_idx]))
        heading_error_deg = normalize_angle_deg(target_heading_deg - yaw_deg)
        return {
            "closest_idx": int(closest_idx),
            "closest_dist_m": float(closest_dist_m),
            "lookahead_m": float(lookahead_m),
            "target_idx": int(target_idx),
            "target_x_m": float(target_x_m),
            "target_y_m": float(target_y_m),
            "target_heading_deg": float(target_heading_deg),
            "heading_error_deg": float(heading_error_deg),
            "curvature": float(curvature),
            "x_vehicle": float(x_vehicle),
            "y_vehicle": float(y_vehicle),
            "target_behind": bool(target_behind),
        }

    def _should_force_resync(self, state, target_behind_counter):
        reasons = []
        if state["closest_dist_m"] > self.resync_dist_m:
            reasons.append("off_path")
        if abs(state["heading_error_deg"]) > self.resync_heading_error_deg:
            reasons.append("heading")
        if target_behind_counter >= self.target_behind_resync_cycles:
            reasons.append("target_behind")
        return reasons

    def _pure_pursuit_steering(self, x_m, y_m, yaw_rad, target_x_m, target_y_m):
        dx = float(target_x_m - x_m)
        dy = float(target_y_m - y_m)

        cos_yaw = math.cos(yaw_rad)
        sin_yaw = math.sin(yaw_rad)

        # Vehicle frame: +x forward, +y left.
        x_vehicle = cos_yaw * dx + sin_yaw * dy
        y_vehicle = -sin_yaw * dx + cos_yaw * dy
        lookahead_sq = x_vehicle * x_vehicle + y_vehicle * y_vehicle

        if lookahead_sq < 1e-6:
            return 0.0, 0.0, x_vehicle, y_vehicle, False

        if x_vehicle <= 0.05:
            steer_norm = (
                0.0 if abs(y_vehicle) < 1e-6 else math.copysign(1.0, y_vehicle)
            )
            return steer_norm, 0.0, x_vehicle, y_vehicle, True

        curvature = 2.0 * y_vehicle / lookahead_sq
        return 0.0, curvature, x_vehicle, y_vehicle, False

    def _curvature_to_steering_cmd(self, curvature):
        if abs(self.steer_max_rad) < 1e-9:
            bicycle_steering_cmd = 0.0
        else:
            bicycle_steering_cmd = math.atan(self.wheelbase_m * curvature) / self.steer_max_rad

        if abs(self.empirical_kappa_slope) < 1e-9:
            empirical_steering_cmd = self.empirical_steering_offset
        else:
            empirical_steering_cmd = (
                self.empirical_steering_offset
                + float(curvature) / self.empirical_kappa_slope
            )

        empirical_steering_cmd = (
            self.empirical_steering_offset
            + self.steering_sign
            * (empirical_steering_cmd - self.empirical_steering_offset)
        )
        bicycle_steering_cmd = self.steering_sign * bicycle_steering_cmd

        if self.curvature_to_steering == "empirical":
            selected_steering_cmd = empirical_steering_cmd
        else:
            selected_steering_cmd = bicycle_steering_cmd

        selected_steering_cmd = float(np.clip(selected_steering_cmd, -1.0, 1.0))
        bicycle_steering_cmd = float(np.clip(bicycle_steering_cmd, -1.0, 1.0))
        empirical_steering_cmd = float(np.clip(empirical_steering_cmd, -1.0, 1.0))
        return selected_steering_cmd, bicycle_steering_cmd, empirical_steering_cmd

    def run(self, x, y, yaw_deg, speed_mps):
        if not (is_valid_number(x) and is_valid_number(y) and is_valid_number(yaw_deg)):
            self.last_debug = None
            return 0, 0.0, None

        x_m = float(x)
        y_m = float(y)
        yaw_deg = float(yaw_deg)
        yaw_rad = math.radians(yaw_deg)
        speed_mps = max(0.0, float(speed_mps)) if is_valid_number(speed_mps) else 0.0

        closest_idx, closest_dist_m = self._find_closest_idx_forward(x_m, y_m)
        state = self._compute_tracking_state(
            x_m,
            y_m,
            yaw_deg,
            yaw_rad,
            speed_mps,
            closest_idx,
            closest_dist_m,
        )
        candidate_target_behind_counter = (
            self.target_behind_counter + 1 if state["target_behind"] else 0
        )
        resync_reasons = self._should_force_resync(
            state, candidate_target_behind_counter
        )
        resynced_this_cycle = False
        if resync_reasons:
            closest_idx, closest_dist_m = self._full_resync_idx(x_m, y_m)
            self.closest_idx = int(closest_idx)
            self.initial_sync_done = True
            state = self._compute_tracking_state(
                x_m,
                y_m,
                yaw_deg,
                yaw_rad,
                speed_mps,
                closest_idx,
                closest_dist_m,
            )
            self.target_behind_counter = (
                self.target_behind_counter + 1 if state["target_behind"] else 0
            )
            resynced_this_cycle = True
        else:
            self.target_behind_counter = candidate_target_behind_counter

        closest_idx = state["closest_idx"]
        closest_dist_m = state["closest_dist_m"]
        lookahead_m = state["lookahead_m"]
        target_idx = state["target_idx"]
        target_x_m = state["target_x_m"]
        target_y_m = state["target_y_m"]
        target_heading_deg = state["target_heading_deg"]
        heading_error_deg = state["heading_error_deg"]
        curvature = state["curvature"]
        x_vehicle = state["x_vehicle"]
        y_vehicle = state["y_vehicle"]
        target_behind = state["target_behind"]
        if target_behind:
            steer_norm = 0.0 if abs(y_vehicle) < 1e-6 else math.copysign(1.0, y_vehicle)
            bicycle_steering_cmd = float(np.clip(self.steering_sign * steer_norm, -1.0, 1.0))
            empirical_steering_cmd = bicycle_steering_cmd
        else:
            steer_norm, bicycle_steering_cmd, empirical_steering_cmd = (
                self._curvature_to_steering_cmd(curvature)
            )

        throttle_erpm = int(self.target_erpm[target_idx])
        if target_behind:
            throttle_erpm = min(throttle_erpm, self.behind_target_erpm)
        if closest_dist_m >= self.off_path_stop_m:
            throttle_erpm = 0
        elif closest_dist_m >= self.off_path_slowdown_m:
            throttle_erpm = min(throttle_erpm, self.off_path_slowdown_erpm)

        throttle_erpm = int(
            np.clip(
                throttle_erpm,
                self.throttle_floor_erpm,
                self.throttle_ceiling_erpm,
            )
        )

        self.last_debug = {
            "reverse_path": bool(self.reverse_path),
            "resynced_this_cycle": bool(resynced_this_cycle),
            "resync_reason": "|".join(resync_reasons),
            "target_behind_counter": int(self.target_behind_counter),
            "resync_dist_m": float(self.resync_dist_m),
            "resync_heading_error_deg": float(self.resync_heading_error_deg),
            "closest_idx": int(closest_idx),
            "target_idx": int(target_idx),
            "closest_dist_m": float(closest_dist_m),
            "lookahead_m": float(lookahead_m),
            "yaw_deg": float(yaw_deg),
            "target_x_m": float(target_x_m),
            "target_y_m": float(target_y_m),
            "target_heading_deg": float(target_heading_deg),
            "heading_error_deg": float(heading_error_deg),
            "target_kappa": float(self.path_kappa[target_idx]),
            "pp_curvature": float(curvature),
            "x_vehicle": float(x_vehicle),
            "y_vehicle": float(y_vehicle),
            "speed_mps": float(speed_mps),
            "target_behind": bool(target_behind),
            "curvature_to_steering_mode": self.curvature_to_steering,
            "empirical_kappa_slope": float(self.empirical_kappa_slope),
            "empirical_steering_offset": float(self.empirical_steering_offset),
            "bicycle_steering_cmd": float(bicycle_steering_cmd),
            "empirical_steering_cmd": float(empirical_steering_cmd),
            "selected_steering_cmd": float(steer_norm),
            "throttle_erpm": int(throttle_erpm),
            "steering_norm": float(steer_norm),
        }

        if self.verbose:
            print(
                "[PurePursuit]",
                self.last_debug,
            )

        pp_debug = dict(self.last_debug) if self.verbose else None
        return throttle_erpm, steer_norm, pp_debug

    def shutdown(self):
        pass
