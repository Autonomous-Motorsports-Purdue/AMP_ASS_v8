#!/usr/bin/env python3
# in: odometry/filtered (Odometry), path_csv (x, y[, eRPM]) | out: cmd/steering (Float64, -1 to 1), cmd/throttle (Float64, eRPM)
import math

import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64

from ros2_parts.orientation import normalize_angle_deg, quaternion_to_yaw

WHEELBASE_M = 1.05
STEER_MAX_RAD = math.radians(25.0)
LOOKAHEAD_TIME_S = 0.6
MIN_LOOKAHEAD_M = 4.0
MAX_LOOKAHEAD_M = 10.0
SEARCH_WINDOW = 80
MAX_RESYNC_DIST_M = 15.0
REJOIN_DIST_M = 1.5
REJOIN_EXIT_DIST_M = 0.75
REJOIN_EXIT_HEADING_DEG = 25.0
REJOIN_LOOKAHEAD_M = 2.5
REJOIN_STEER_GAIN = 1.25
RESYNC_HEADING_ERROR_DEG = 120.0
TARGET_BEHIND_RESYNC_CYCLES = 3
THROTTLE_CEILING_ERPM = 4500
BEHIND_TARGET_ERPM = 1200
OFF_PATH_SLOWDOWN_M = 1.5
OFF_PATH_STOP_M = 3.0
OFF_PATH_SLOWDOWN_ERPM = 1200
EMPIRICAL_KAPPA_SLOPE = -0.223  # curvature per unit steering, measured on the kart
EMPIRICAL_STEERING_OFFSET = 0.097  # steering command that drives straight


def load_path(path_csv, fallback_erpm, reverse_path):
    raw = np.atleast_2d(np.genfromtxt(path_csv, delimiter=",", dtype=float, encoding="utf-8"))
    if not np.isfinite(raw[0]).all():  # header row
        raw = np.atleast_2d(np.genfromtxt(path_csv, delimiter=",", dtype=float, encoding="utf-8", skip_header=1))
    if reverse_path:
        raw = raw[::-1].copy()
    xy = raw[:, :2]

    # eRPM column: always the last of 4+, or a 3rd column that's clearly eRPM (not a -1..1 throttle)
    erpm = np.full(len(xy), fallback_erpm, dtype=int)
    if raw.shape[1] >= 4:
        erpm = np.rint(raw[:, -1]).astype(int)
    elif raw.shape[1] == 3:
        third = np.abs(raw[:, 2][np.isfinite(raw[:, 2])])
        if third.size and (third.max() >= 50.0 or np.median(third) >= 25.0):
            erpm = np.rint(raw[:, 2]).astype(int)

    # heading at each point from its neighbours (wraps, since tracks are loops)
    prev_xy, next_xy = np.roll(xy, 1, axis=0), np.roll(xy, -1, axis=0)
    d = next_xy - prev_xy
    degenerate = np.hypot(d[:, 0], d[:, 1]) < 1e-6
    d[degenerate] = (next_xy - xy)[degenerate]
    psi = np.arctan2(d[:, 1], d[:, 0])
    return xy[:, 0], xy[:, 1], psi, erpm


class PurePursuitControllerNode(Node):
    def __init__(self):
        super().__init__("pure_pursuit_controller")
        path_csv = self.declare_parameter("path_csv", "").value
        fallback_erpm = self.declare_parameter("fallback_erpm", 1500).value
        reverse_path = self.declare_parameter("reverse_path", False).value
        self.curvature_to_steering = self.declare_parameter("curvature_to_steering", "empirical").value
        self.steering_sign = self.declare_parameter("steering_sign", 1.0).value

        self.path_x, self.path_y, self.path_psi, self.target_erpm = load_path(path_csv, fallback_erpm, reverse_path)

        self.closest_idx = None
        self.rejoin_active = False
        self.target_behind_counter = 0

        self.create_subscription(Odometry, "odometry/filtered", self.run, 10)
        self.steer_pub = self.create_publisher(Float64, "cmd/steering", 10)
        self.throt_pub = self.create_publisher(Float64, "cmd/throttle", 10)

    def full_resync(self, x, y):
        distances = np.hypot(self.path_x - x, self.path_y - y)
        self.closest_idx = int(np.argmin(distances))
        return self.closest_idx, float(distances[self.closest_idx])

    def find_closest_forward(self, x, y):
        # search a window ahead of last time; full search on startup or if we're far from it
        if self.closest_idx is None or math.hypot(x - self.path_x[self.closest_idx],
                                                  y - self.path_y[self.closest_idx]) > MAX_RESYNC_DIST_M:
            return self.full_resync(x, y)
        window = (self.closest_idx + np.arange(SEARCH_WINDOW + 1)) % len(self.path_x)
        distances = np.hypot(self.path_x[window] - x, self.path_y[window] - y)
        best = int(np.argmin(distances))
        self.closest_idx = int(window[best])
        return self.closest_idx, float(distances[best])

    def pick_lookahead_target(self, closest_idx, lookahead_m):
        # walk along the path lookahead_m meters and interpolate within the segment
        n = len(self.path_x)
        i = closest_idx
        for _ in range(n + 2):
            j = (i + 1) % n
            seg_dx = self.path_x[j] - self.path_x[i]
            seg_dy = self.path_y[j] - self.path_y[i]
            seg_len = math.hypot(seg_dx, seg_dy)
            if seg_len >= 1e-6:
                if lookahead_m <= seg_len:
                    ratio = lookahead_m / seg_len
                    return j, self.path_x[i] + ratio * seg_dx, self.path_y[i] + ratio * seg_dy
                lookahead_m -= seg_len
            i = j
        return closest_idx, self.path_x[closest_idx], self.path_y[closest_idx]

    def track(self, x, y, yaw_deg, speed, closest_idx, rejoin):
        lookahead_m = float(np.clip(speed * LOOKAHEAD_TIME_S, MIN_LOOKAHEAD_M, MAX_LOOKAHEAD_M))
        if rejoin:
            lookahead_m = min(lookahead_m, REJOIN_LOOKAHEAD_M)
        target_idx, tx, ty = self.pick_lookahead_target(closest_idx, lookahead_m)

        # target in the vehicle frame: +x forward, +y left
        yaw = math.radians(yaw_deg)
        dx, dy = tx - x, ty - y
        x_v = math.cos(yaw) * dx + math.sin(yaw) * dy
        y_v = -math.sin(yaw) * dx + math.cos(yaw) * dy
        lookahead_sq = x_v * x_v + y_v * y_v
        behind = lookahead_sq >= 1e-6 and x_v <= 0.05
        curvature = 2.0 * y_v / lookahead_sq if lookahead_sq >= 1e-6 and not behind else 0.0

        heading_error = normalize_angle_deg(math.degrees(self.path_psi[target_idx]) - yaw_deg)
        return dict(target_idx=target_idx, y_v=y_v, behind=behind, curvature=curvature,
                    heading_error=heading_error, rejoin=rejoin)

    def track_with_rejoin(self, x, y, yaw_deg, speed, closest_idx, closest_dist):
        # rejoin starts on distance alone, and only ends once close to the path and pointed along it
        if not self.rejoin_active:
            return self.track(x, y, yaw_deg, speed, closest_idx, closest_dist > REJOIN_DIST_M)
        state = self.track(x, y, yaw_deg, speed, closest_idx, True)
        if closest_dist < REJOIN_EXIT_DIST_M and abs(state["heading_error"]) < REJOIN_EXIT_HEADING_DEG:
            state = self.track(x, y, yaw_deg, speed, closest_idx, False)
        return state

    def curvature_to_steering_cmd(self, curvature):
        if self.curvature_to_steering == "bicycle":
            cmd = self.steering_sign * math.atan(WHEELBASE_M * curvature) / STEER_MAX_RAD
        else:
            cmd = EMPIRICAL_STEERING_OFFSET + self.steering_sign * curvature / EMPIRICAL_KAPPA_SLOPE
        return float(np.clip(cmd, -1.0, 1.0))

    def run(self, odom):
        x, y = odom.pose.pose.position.x, odom.pose.pose.position.y
        yaw_deg = math.degrees(quaternion_to_yaw(odom.pose.pose.orientation))
        speed = max(0.0, odom.twist.twist.linear.x)

        closest_idx, closest_dist = self.find_closest_forward(x, y)
        state = self.track_with_rejoin(x, y, yaw_deg, speed, closest_idx, closest_dist)

        # resync to the nearest point on the whole path if we're pointed way off or the target keeps ending up behind us
        behind_count = self.target_behind_counter + 1 if state["behind"] else 0
        if abs(state["heading_error"]) > RESYNC_HEADING_ERROR_DEG or behind_count >= TARGET_BEHIND_RESYNC_CYCLES:
            closest_idx, closest_dist = self.full_resync(x, y)
            state = self.track_with_rejoin(x, y, yaw_deg, speed, closest_idx, closest_dist)
            behind_count = self.target_behind_counter + 1 if state["behind"] else 0
        self.target_behind_counter = behind_count
        self.rejoin_active = state["rejoin"]

        if state["behind"]:
            # turn hard toward whichever side the target is on
            steer = 0.0 if abs(state["y_v"]) < 1e-6 else math.copysign(1.0, state["y_v"])
        else:
            steer = self.curvature_to_steering_cmd(state["curvature"])
            if state["rejoin"]:
                steer = float(np.clip(steer * REJOIN_STEER_GAIN, -1.0, 1.0))

        throttle = int(self.target_erpm[state["target_idx"]])
        if state["behind"]:
            throttle = min(throttle, BEHIND_TARGET_ERPM)
        if closest_dist >= OFF_PATH_STOP_M:
            throttle = 0
        elif closest_dist >= OFF_PATH_SLOWDOWN_M:
            throttle = min(throttle, OFF_PATH_SLOWDOWN_ERPM)
        throttle = int(np.clip(throttle, 0, THROTTLE_CEILING_ERPM))

        self.steer_pub.publish(Float64(data=steer))
        self.throt_pub.publish(Float64(data=float(throttle)))


def main(args=None):
    rclpy.init(args=args)
    node = PurePursuitControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
