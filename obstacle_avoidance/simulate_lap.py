import math
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

from kinematic_model import KinematicBicycleModel
from pure_pursuit_obstacle_controller import PurePursuitController
from kartRRT import KartRRT, inflate

def generate_synthetic_obstacle(obs_x, obs_y, radius=0.8, num_pts=120):
    angles = np.linspace(0, 2 * math.pi, num_pts)
    r = radius + np.random.uniform(-0.02, 0.02, size=num_pts)
    x = obs_x + r * np.cos(angles)
    y = obs_y + r * np.sin(angles)
    return np.column_stack([x, y])


def points_to_grid(points, veh_x, veh_y, veh_yaw, grid_res=0.2, grid_range=14.0):
    grid_size = int((grid_range * 2) / grid_res)
    grid = np.zeros((grid_size, grid_size), dtype=bool)

    if len(points) == 0:
        return grid, grid_res

    cos_yaw, sin_yaw = math.cos(veh_yaw), math.sin(veh_yaw)
    dx = points[:, 0] - veh_x
    dy = points[:, 1] - veh_y

    local_x = cos_yaw * dx + sin_yaw * dy
    local_y = -sin_yaw * dx + cos_yaw * dy

    gx = np.clip(((local_x + grid_range) / grid_res).astype(np.int32), 0, grid_size - 1)
    gy = np.clip(((local_y + grid_range) / grid_res).astype(np.int32), 0, grid_size - 1)
    grid[gy, gx] = True

    return grid, grid_res


def check_direct_collision(path_x, path_y, obstacle_pts, safety_dist=2.8):
    for px, py in zip(path_x, path_y):
        dists = np.hypot(obstacle_pts[:, 0] - px, obstacle_pts[:, 1] - py)
        if np.min(dists) < safety_dist:
            return True
    return False


def simulate_pure_pursuit_lap(raceline_csv):
    dt_sim = 0.02
    max_sim_time = 240.0
    nominal_speed_mps = 5.0
    veh_radius = 2.2
    grid_res = 0.2
    grid_range = 14.0

    pp_ctrl = PurePursuitController(path_csv=raceline_csv, wheelbase_m=1.0005)

    base_x = np.copy(pp_ctrl.path_x_m)
    base_y = np.copy(pp_ctrl.path_y_m)
    base_psi = np.copy(pp_ctrl.path_psi_rad)
    n_pts = len(base_x)

    # Initial start alignment with vehicle rear axle
    start_x, start_y, start_heading = base_x[0], base_y[0], base_psi[0]
    cam_offset = 0.6096
    axle_start_x = start_x - cam_offset * math.cos(start_heading)
    axle_start_y = start_y - cam_offset * math.sin(start_heading)

    vehicle = KinematicBicycleModel(axle_start_x, axle_start_y, start_heading, wheelbase=1.0005, dt=dt_sim)

    candidate_indices = [
        int(n_pts * 0.25),
        int(n_pts * 0.55),
        int(n_pts * 0.80)
    ]

    obstacle_centers = []
    all_obstacle_pts = []
    for idx in candidate_indices:
        obs_x_cand, obs_y_cand = base_x[idx], base_y[idx]
        if math.hypot(obs_x_cand - start_x, obs_y_cand - start_y) > 15.0:
            obstacle_centers.append((obs_x_cand, obs_y_cand, idx))
            pts = generate_synthetic_obstacle(obs_x_cand, obs_y_cand, radius=0.8)
            all_obstacle_pts.append(pts)

    obstacle_pts = np.vstack(all_obstacle_pts)

    traj_x, traj_y = [vehicle.x], [vehicle.y]
    lap_done = False
    lap_time = None
    rrt_call_count = 0
    collision_count = 0

    detour_active = False
    detour_end_idx = -1
    detour_offsets = {}  # Index-mapped lateral offsets

    print(f"[SIMULATION] Running Robust Pure Pursuit + KartRRT Engine...")

    t_start = time.time()

    for t in np.arange(0, max_sim_time, dt_sim):
        if lap_done:
            break

        # 1. PHYSICAL COLLISION MONITORING
        dist_to_nearest_obs = np.min(np.hypot(obstacle_pts[:, 0] - vehicle.x, obstacle_pts[:, 1] - vehicle.y))
        if dist_to_nearest_obs < 0.8:
            collision_count += 1
            print(f"[{t:.2f}s] COLLISION ALERT! Kart at ({vehicle.x:.2f}, {vehicle.y:.2f}) - Clearance: {dist_to_nearest_obs:.2f}m")

        closest_idx, _ = pp_ctrl.find_closest_idx_forward(vehicle.x, vehicle.y)

        # 2. HORIZON COLLISION DETECTION
        lookahead_step = 12
        lookahead_idx = (closest_idx + lookahead_step) % n_pts

        if lookahead_idx > closest_idx:
            horizon_x = base_x[closest_idx:lookahead_idx]
            horizon_y = base_y[closest_idx:lookahead_idx]
        else:
            horizon_x = np.concatenate([base_x[closest_idx:], base_x[:lookahead_idx]])
            horizon_y = np.concatenate([base_y[closest_idx:], base_y[:lookahead_idx]])

        blocked = check_direct_collision(horizon_x, horizon_y, obstacle_pts, safety_dist=2.8)

        # 3. KINEMATIC RRT PLANNER TRIGGER
        if blocked and not detour_active:
            rrt_call_count += 1
            detour_active = True

            dists = [math.hypot(c[0] - vehicle.x, c[1] - vehicle.y) for c in obstacle_centers]
            target_obs = obstacle_centers[np.argmin(dists)]
            obs_idx = target_obs[2]

            # Reconnection window: 30 points (~6.0m) downstream
            start_detour_idx = closest_idx
            detour_end_idx = (obs_idx + 30) % n_pts

            goal_x, goal_y = base_x[detour_end_idx], base_y[detour_end_idx]
            goal_heading = base_psi[detour_end_idx]

            # Calculate outward shift vector
            psi_goal = goal_heading
            nx, ny = -math.sin(psi_goal), math.cos(psi_goal)
            obs_vec_x, obs_vec_y = target_obs[0] - goal_x, target_obs[1] - goal_y
            dodge_dir = -1.0 if (obs_vec_x * nx + obs_vec_y * ny) > 0 else 1.0

            # Pass unblocked local goal to RRT
            goal_x_offset = goal_x + nx * 2.2 * dodge_dir
            goal_y_offset = goal_y + ny * 2.2 * dodge_dir

            raw_grid, res = points_to_grid(obstacle_pts, vehicle.x, vehicle.y, vehicle.heading, grid_res, grid_range)
            inflated_map = inflate(raw_grid, res, max_radius=veh_radius)

            planner = KartRRT(
                grid=inflated_map,
                resolution=res,
                wheelbase=vehicle.wheelbase,
                speed=nominal_speed_mps,
                iters=400,
                sample_rate=0.45,
                goal_xy_threshold=1.5,
                goal_yaw_threshold=math.radians(180),
            )

            cos_y, sin_y = math.cos(vehicle.heading), math.sin(vehicle.heading)
            dx, dy = goal_x_offset - vehicle.x, goal_y_offset - vehicle.y
            local_goal_x = cos_y * dx + sin_y * dy
            local_goal_y = -sin_y * dx + cos_y * dy

            rrt_path, _, info = planner.plan(
                start=(grid_range, grid_range, 0.0),
                goal=(local_goal_x + grid_range, local_goal_y + grid_range, 0.0),
            )

            # Build smooth index-mapped offset profile
            detour_offsets = {}
            total_detour_len = (detour_end_idx - start_detour_idx + n_pts) % n_pts
            
            for i in range(total_detour_len):
                curr_i = (start_detour_idx + i) % n_pts
                progress = i / float(total_detour_len)
                # Cosine bell envelope peaking at 2.4m clearance near obstacle center
                detour_offsets[curr_i] = 2.4 * math.sin(progress * math.pi) * dodge_dir

        # 4. RELEASE DETOUR MODE WHEN PAST RECONNECTION POINT
        if detour_active and closest_idx == detour_end_idx:
            detour_active = False

        # 5. TARGET SELECTION (With Smooth Lateral Envelope)
        track_lookahead_step = 8
        target_idx = (closest_idx + track_lookahead_step) % n_pts

        target_x = base_x[target_idx]
        target_y = base_y[target_idx]

        if detour_active and target_idx in detour_offsets:
            offset = detour_offsets[target_idx]
            psi_target = base_psi[target_idx]
            nx, ny = -math.sin(psi_target), math.cos(psi_target)
            target_x += nx * offset
            target_y += ny * offset

        # 6. PURE PURSUIT CONTROL STEP
        dx = target_x - vehicle.x
        dy = target_y - vehicle.y
        yaw_rad = vehicle.heading

        cos_yaw, sin_yaw = math.cos(yaw_rad), math.sin(yaw_rad)
        x_veh = cos_yaw * dx + sin_yaw * dy
        y_veh = -sin_yaw * dx + cos_yaw * dy
        lookahead_sq = x_veh**2 + y_veh**2

        if lookahead_sq > 1e-6:
            curvature = 2.0 * y_veh / lookahead_sq
            bicycle_steer = math.atan(vehicle.wheelbase * curvature) / pp_ctrl.steer_max_rad
            steer_norm = float(np.clip(bicycle_steer, -1.0, 1.0))
        else:
            steer_norm = 0.0

        vehicle.update(nominal_speed_mps, steer_norm)
        traj_x.append(vehicle.x)
        traj_y.append(vehicle.y)

        # 7. LAP COMPLETION DETECTOR
        dist_to_start = math.hypot(vehicle.x - start_x, vehicle.y - start_y)
        if t > 15.0 and dist_to_start < 3.5 and closest_idx > n_pts - 20:
            lap_done = True
            lap_time = t

    exec_time = time.time() - t_start

    print(f"\n================ SIMULATION RESULTS ================")
    print(f"Computation Time:  {exec_time:.2f} seconds")
    print(f"RRT Planner Calls: {rrt_call_count} times")
    print(f"Total Collisions:  {collision_count} frames")
    if lap_done:
        print(f"Simulated Lap Time: {lap_time:.2f} seconds")
    else:
        print(f"Status: Incomplete (Timeout reached)")
    print(f"===================================================\n")

    plt.figure(figsize=(10, 6))
    plt.plot(base_x, base_y, 'r--', label='Target Raceline')
    plt.plot(traj_x, traj_y, 'g-', linewidth=2, label='Pure Pursuit Trajectory')
    plt.plot(obstacle_pts[:, 0], obstacle_pts[:, 1], 'k.', markersize=6, label='LiDAR Obstacles')
    plt.scatter(start_x, start_y, color='blue', marker='o', s=80, label='Start / Finish')

    if lap_done:
        plt.text(traj_x[-1], traj_y[-1], f"  Lap Time: {lap_time:.2f}s", fontsize=11, color='green', fontweight='bold')

    plt.title("Pure Pursuit + KartRRT Obstacle Avoidance Simulation")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    raceline_file = "gps_paths/grand_prix_xy.csv"
    try:
        pd.read_csv(raceline_file)
    except Exception:
        s = np.linspace(0, 40, 200)
        x = s
        y = 2.0 * np.sin(s / 3.0)
        df = pd.DataFrame({'x': x, 'y': y})
        df.to_csv(raceline_file, index=False)

    simulate_pure_pursuit_lap(raceline_file)