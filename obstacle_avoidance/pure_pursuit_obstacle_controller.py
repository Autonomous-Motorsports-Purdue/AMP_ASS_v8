import math
import numpy as np
import pandas as pd

def normalize_angle_rad(angle_rad):
    return (float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi


def normalize_angle_deg(angle_deg):
    return (float(angle_deg) + 180.0) % 360.0 - 180.0


class PurePursuitController:
    """Pure Pursuit with Obstacle Avoidance"""

    def __init__(
        self,
        path_csv,
        wheelbase_m=1,
        steer_max_deg=30.0,
        lookahead_time_s=0.6,
        min_lookahead_m=2.0,
        max_lookahead_m=6.0,
        fallback_erpm=1500,
        steering_sign=1.0,
    ):
        line = self._load_line(path_csv, fallback_erpm)
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
        self.steering_sign = float(steering_sign)
        self.closest_idx = 0

    @classmethod
    def _load_line(cls, path_csv, fallback_erpm):
        try:
            df = pd.read_csv(path_csv)
            df.columns = df.columns.str.strip().str.lower()
            if 'x' not in df.columns and 'x_m' not in df.columns:
                df = pd.read_csv(path_csv, sep=';', skiprows=2)
                df.columns = df.columns.str.strip().str.lower()
        except Exception:
            df = pd.read_csv(path_csv, sep=';', skiprows=2)
            df.columns = df.columns.str.strip().str.lower()

        df.rename(columns={'x_m': 'x', 'y_m': 'y'}, inplace=True)

        x = df['x'].to_numpy(dtype=float)
        y = df['y'].to_numpy(dtype=float)
        n = len(x)

        if 'psi' in df.columns:
            psi = df['psi'].to_numpy(dtype=float)
        elif 'psi_rad' in df.columns:
            psi = df['psi_rad'].to_numpy(dtype=float)
        else:
            psi = np.zeros(n)
            for i in range(n):
                next_i = (i + 1) % n
                prev_i = (i - 1) % n
                psi[i] = math.atan2(y[next_i] - y[prev_i], x[next_i] - x[prev_i])

        seg_lengths = np.hypot(np.diff(x), np.diff(y))
        s_m = np.concatenate(([0.0], np.cumsum(seg_lengths)))
        kappa = np.zeros(n)
        erpm = np.full(n, int(fallback_erpm), dtype=int)

        return np.column_stack([s_m, x, y, psi, kappa, erpm])

    def find_closest_idx_forward(self, x_m, y_m):
        distances = np.hypot(self.path_x_m - x_m ,self.path_y_m - y_m)
        self.closest_idx = int(np.argmin(distances))
        return self.closest_idx, float(distances[self.closest_idx])


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

    def run(self, x, y, yaw_deg, speed_mps):
        x_m, y_m = float(x), float(y)
        yaw_rad = math.radians(float(yaw_deg))
        speed_mps = max(0.0, float(speed_mps))

        closest_idx, _ = self.find_closest_idx_forward(x_m, y_m)
        lookahead_m = float(np.clip(speed_mps * self.lookahead_time_s, self.min_lookahead_m, self.max_lookahead_m))
        target_idx, target_x, target_y = self._pick_lookahead_target(closest_idx, lookahead_m)

        dx, dy = target_x - x_m, target_y - y_m
        cos_yaw, sin_yaw = math.cos(yaw_rad), math.sin(yaw_rad)

        x_veh = cos_yaw * dx + sin_yaw * dy
        y_veh = -sin_yaw * dx + cos_yaw * dy
        lookahead_sq = x_veh ** 2 + y_veh ** 2

        if lookahead_sq < 1e-6:
            return int(self.target_erpm[target_idx]), 0.0, None

        curvature = 2.0 * y_veh / lookahead_sq
        bicycle_steer = math.atan(self.wheelbase_m * curvature) / self.steer_max_rad
        steer_norm = float(np.clip(self.steering_sign * bicycle_steer, -1.0, 1.0))
        throttle_erpm = int(self.target_erpm[target_idx])

        return throttle_erpm, steer_norm, None