import argparse
import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
from pipeline import ObstacleAwarePursuit


TRACK_CSV = "gps_paths/grand_prix_xy.csv"
SPEED = 6.0      

# Obstacles as (distance along the lap in m, lateral offset in m, + = left).
OBSTACLES = [
    (60, 0.0),
    (130, 0.8),
    (200, -0.8),
    (230, 0.0),
    (270, 0.0),
    (340, 1.0),
    (360, 0.0),
    (361, 1.0),
    (410, -1.0),
]
# Radius of a large traffic cone in meters
OBSTACLE_RADIUS = 0.2 
# --------------------------------------------------------------------------

LIDAR_RANGE = 6.0       # Meters
KART_RADIUS = 0.5       # safety circle around the kart centre (m)
TRACK_HALF_WIDTH = 2.5  # corridor used by both the sim (off-track check) and the planner
STALL_TIMEOUT = 3.0     # if no path found in this seconds, end run

@dataclass
# Circular obstacle on the track frame meters
class Obstacle:
    x: float
    y: float
    radius: float = OBSTACLE_RADIUS


def place_on_reference(ref, s, lateral=0.0):
    """Position at arc length s (m) along the reference lap, offset laterally (+ = left)."""
    s = s % ref.ref_len
    i = int(np.searchsorted(ref.ref_s, s)) % len(ref.ref_x)
    psi = ref.ref_psi[i]
    return (float(ref.ref_x[i] - lateral * math.sin(psi)),
            float(ref.ref_y[i] + lateral * math.cos(psi)))


def lidar_points(obstacles, x, y, yaw):
    """Sensor-frame points on the boundary of each obstacle within LiDAR range."""
    # Simulate a LiDAR scan of the cones
    c, s = math.cos(yaw), math.sin(yaw)
    ang = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    out = []
    for ob in obstacles:
        ring = np.column_stack([ob.x + ob.radius * np.cos(ang),
                                ob.y + ob.radius * np.sin(ang)])
        rel = ring - np.array([x, y])
        sx = c * rel[:, 0] + s * rel[:, 1]
        sy = -s * rel[:, 0] + c * rel[:, 1]
        pts = np.column_stack([sx, sy])

        # Keep points within sensor range
        out.append(pts[np.hypot(pts[:, 0], pts[:, 1]) < LIDAR_RANGE])
    if not out:
        return np.zeros((0, 2))
    return np.vstack(out)


def run_episode(pipe, obstacles, speed, dt=0.05, max_time_s=260.0):
    # Simulate one lap around the track
    # Read LiDAR -> GEt Steering Command -> Apply to Bicycle Model -> Check For Collision / Finished Lap
    ref_tree = cKDTree(np.column_stack([pipe.ref_x, pipe.ref_y]))

    x, y, yaw = pipe.ref_x[0], pipe.ref_y[0], pipe.ref_psi[0]
    steer_actual = 0.0
    max_steer = math.radians(30)
    max_rate = math.radians(90)
    tau = 0.15       # steering actuator lag (s)
    wheelbase = 1.0  # meters

    t = 0.0 # Simulation time seconds
    progress = 0.0   # reference arc covered (m)
    last_ref_idx = ref_tree.query([x, y])[1]
    lap_time = None
    stalled = False
    blocked_for = 0.0
    trace = [(x, y)]

    while t < max_time_s:
        # Get LiDAR points
        pts = lidar_points(obstacles, x, y, yaw)

        # Plan and control
        _erpm, steer_cmd, status = pipe.step(x, y, math.degrees(yaw), speed, pts, t)

        # Check if path is blocked
        if status == "blocked_no_path":
            v, steer_target = 0.0, 0.0
            blocked_for += dt
            if blocked_for > STALL_TIMEOUT:
                stalled = True
                break
        else:
            v, steer_target = speed, steer_cmd * max_steer
            blocked_for = 0.0

        # actuator: rate limit + first-order lag
        err = steer_target - steer_actual
        steer_actual += float(np.clip(err / tau, -max_rate, max_rate)) * dt
        steer_actual = float(np.clip(steer_actual, -max_steer, max_steer))

        # kinematic bicycle update
        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
        yaw += (v / wheelbase) * math.tan(steer_actual) * dt
        yaw = math.atan2(math.sin(yaw), math.cos(yaw))

        trace.append((x, y))
        t += dt

        # How far along the reference lap the kart has moved
        xte, ridx = ref_tree.query([x, y])
        progress += (pipe.ref_s[ridx] - pipe.ref_s[last_ref_idx]) % pipe.ref_len
        last_ref_idx = ridx

        # Collision check
        collided = any(math.hypot(x - ob.x, y - ob.y) < KART_RADIUS + ob.radius
                       for ob in obstacles)
        if collided or xte > TRACK_HALF_WIDTH:
            break
        if progress >= pipe.ref_len:
            lap_time = t
            break

    return trace, lap_time, stalled


def plot_episode(pipe, obstacles, trace, lap_time, stalled, speed):
    import matplotlib.pyplot as plt
    tr = np.array(trace)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(pipe.ref_x, pipe.ref_y, "--", color="gray", lw=1, label="reference lap")
    ax.plot(tr[:, 0], tr[:, 1], color="tab:blue", lw=1.5, label="kart trajectory")
    for ob in obstacles:
        ax.add_patch(plt.Circle((ob.x, ob.y), ob.radius, color="red", alpha=0.6))
    ax.plot(tr[0, 0], tr[0, 1], "go", label="start")
    ax.set_aspect("equal")

    if lap_time is not None:
        lap_text = f"lap time {lap_time:.1f} s"
    elif stalled:
        lap_text = "stalled (no detour)"
    else:
        lap_text = "lap not completed"
    
    ax.set_title(f"{len(obstacles)} obstacles  |  {speed:.1f} m/s  |  {lap_text}")
    ax.legend()
    plt.show()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Closed-loop obstacle-avoidance simulation.")
    parser.add_argument("--speed", type=float, default=SPEED, help="kart speed (m/s)")
    args = parser.parse_args(argv)

    pipe = ObstacleAwarePursuit(TRACK_CSV, track_half_width=TRACK_HALF_WIDTH)
    obstacles = [Obstacle(*place_on_reference(pipe, s, lat)) for s, lat in OBSTACLES]

    trace, lap_time, stalled = run_episode(pipe, obstacles, args.speed)
    plot_episode(pipe, obstacles, trace, lap_time, stalled, args.speed)


if __name__ == "__main__":
    main()