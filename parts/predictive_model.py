import json
import math
import time
from pathlib import Path

import numpy as np

R_EARTH = 6378137.0
RAD2DEG = 180.0 / np.pi

FEATURE_NAMES = [
    "vdt_cos_gps_yaw",
    "vdt_sin_gps_yaw",
    "vdt_cos_imu_yaw",
    "vdt_sin_imu_yaw",
]

# Shared blend weights applied to GPS and IMU velocity components.
BLEND_FEATURE_NAMES = ["gps_motion", "imu_motion"]


def latlon_to_xy(lat_deg, lon_deg, ref_lat_deg, ref_lon_deg):
    cos_ref = max(math.cos(math.radians(ref_lat_deg)), 1e-6)
    y_north = (float(lat_deg) - ref_lat_deg) / RAD2DEG * R_EARTH
    x_east = (float(lon_deg) - ref_lon_deg) / RAD2DEG * (R_EARTH * cos_ref)
    return x_east, y_north


def compass_heading_to_math_yaw_deg(heading_deg):
    # BNO086 heading is compass-style: 0 north, 90 east, clockwise positive.
    # Math yaw: 0 east, 90 north, counterclockwise positive.
    return (90.0 - float(heading_deg) + 180.0) % 360.0 - 180.0


def build_step_features(
    gps_speed_mps,
    dt,
    gps_heading_deg=None,
    imu_heading_deg=None,
    gps_yaw_math_deg=None,
):
    """
    Interpretable one-step motion features:
      delta_x ~= sum(w_i * feature_i)
      delta_y ~= sum(w_j * feature_j)

    Each feature is speed * direction_component * dt.

    Pass either gps_heading_deg (compass) or gps_yaw_math_deg (already converted).
    """
    dt = max(float(dt), 1e-4)
    speed = float(gps_speed_mps or 0.0)

    if imu_heading_deg is None:
        raise ValueError("imu_heading_deg is required")

    imu_yaw = math.radians(compass_heading_to_math_yaw_deg(imu_heading_deg))
    if gps_yaw_math_deg is not None:
        gps_yaw = math.radians(float(gps_yaw_math_deg))
    elif gps_heading_deg is not None:
        gps_yaw = math.radians(compass_heading_to_math_yaw_deg(gps_heading_deg))
    else:
        gps_yaw = imu_yaw

    return np.array(
        [
            speed * math.cos(gps_yaw) * dt,
            speed * math.sin(gps_yaw) * dt,
            speed * math.cos(imu_yaw) * dt,
            speed * math.sin(imu_yaw) * dt,
        ],
        dtype=float,
    )

class TinyPredictiveModel:
    def __init__(self):
        pass

    def run(self, x, y, gps_speed, gps_yaw, dt):
        inputs = (x, y, gps_speed, gps_yaw)
        if any([i is None for i in inputs]):
            return x, y
        else:
            x_f = x + gps_speed * np.cos(np.deg2rad(gps_yaw)) * dt
            y_f = y + gps_speed * np.sin(np.deg2rad(gps_yaw)) * dt
            return x_f, y_f

class PredictiveModel:
    """
    Predict next (x, y) from current position and GPS+IMU motion features.

    Baseline (no trained weights):
      x_next = x + speed * cos(gps_yaw) * dt
      y_next = y + speed * sin(gps_yaw) * dt

    Trained mode learns weights on the four v*dt*cos/sin terms above.
    """

    def __init__(self, weights=None, model_path=None, time_func=time.monotonic):
        self.time_func = time_func
        self.timer = None
        self.weights = None

        if model_path is not None:
            self.load(model_path)
        elif weights is not None:
            self.weights = np.asarray(weights, dtype=float)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        model = cls(weights=payload["weights"])
        model.model_type = payload.get("model_type", "full_linear")
        model.feature_names = payload.get("feature_names", FEATURE_NAMES)
        model.description = payload.get("description", "")
        return model

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "feature_names": FEATURE_NAMES,
            "weights": self.weights.tolist(),
            "description": (
                "Linear one-step predictor: delta_xy = W @ "
                "[vdt_cos_gps, vdt_sin_gps, vdt_cos_imu, vdt_sin_imu]"
            ),
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    def _predict_delta(
        self,
        gps_speed,
        gps_heading_deg=None,
        imu_heading_deg=None,
        gps_yaw_math_deg=None,
        dt=0.02,
    ):
        features = build_step_features(
            gps_speed,
            dt,
            gps_heading_deg=gps_heading_deg,
            imu_heading_deg=imu_heading_deg,
            gps_yaw_math_deg=gps_yaw_math_deg,
        )
        if self.weights is None:
            return float(features[0]), float(features[1])

        if self.weights.shape[0] == 2:
            # Shared GPS/IMU blend: same two weights for x and y.
            w_gps, w_imu = self.weights[:, 0]
            dx = w_gps * features[0] + w_imu * features[2]
            dy = w_gps * features[1] + w_imu * features[3]
            return float(dx), float(dy)

        delta = features @ self.weights
        return float(delta[0]), float(delta[1])

    def run(
        self,
        x_t,
        y_t,
        gps_speed,
        gps_heading_deg=None,
        imu_heading_deg=None,
        gps_yaw_math_deg=None,
        dt=None,
    ):
        inputs = (x_t, y_t, gps_speed)
        if any(v is not None and not np.isfinite(float(v)) for v in inputs):
            return x_t, y_t

        if dt is None:
            now = self.time_func()
            if self.timer is None:
                self.timer = now
                return x_t, y_t
            dt = now - self.timer
            self.timer = now
        else:
            dt = float(dt)

        if imu_heading_deg is None and gps_heading_deg is None and gps_yaw_math_deg is None:
            return x_t, y_t

        if imu_heading_deg is None:
            imu_heading_deg = gps_heading_deg if gps_heading_deg is not None else 0.0

        dx, dy = self._predict_delta(
            gps_speed,
            gps_heading_deg=gps_heading_deg,
            imu_heading_deg=imu_heading_deg,
            gps_yaw_math_deg=gps_yaw_math_deg,
            dt=dt,
        )
        return float(x_t) + dx, float(y_t) + dy

    def shutdown(self):
        pass
