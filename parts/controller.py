import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
import time

# ================= Kinematic Model =================
class KinematicBicycleModel:
    def __init__(self, x, y, heading, wheelbase, dt):
        self.x = x
        self.y = y
        self.heading = heading
        self.wheelbase = wheelbase
        self.dt = dt

    def update(self, speed, angular_velocity):
        self.x += speed * math.cos(self.heading) * self.dt
        self.y += speed * math.sin(self.heading) * self.dt
        self.heading += angular_velocity * self.dt
        self.heading = (self.heading + math.pi) % (2*math.pi) - math.pi

# ================= Bicycle Model (steering command input) =================
class BicycleModel:
    """
    Kinematic bicycle driven directly by the normalized steering command (-1..1).
    curvature = kappa_slope * (cmd - steering_offset)   ("linear", fitted from logs)
    curvature = -tan(cmd * max_steer_rad) / wheelbase   ("tan", textbook)
    Defaults fitted on 2026-05-19-10-58-55 (pure pursuit, RTK FIXED) with sim/model_check.py.
    The servo responds `delay` seconds after a command; rollout() expects cmd_seq already
    shifted by that delay (in an MPC, the first delay/dt steps are the commands already sent).
    """
    def __init__(self, wheelbase=1.000506, kappa_slope=-0.542, steering_offset=0.063,
                 delay=0.11, mapping="linear", max_steer_rad=0.5):
        self.wheelbase = wheelbase
        self.kappa_slope = kappa_slope
        self.steering_offset = steering_offset
        self.delay = delay
        self.mapping = mapping
        self.max_steer_rad = max_steer_rad

    def curvature(self, cmd):
        cmd = np.asarray(cmd, dtype=float)
        if self.mapping == "tan":
            return -np.tan(cmd * self.max_steer_rad) / self.wheelbase
        return self.kappa_slope * (cmd - self.steering_offset)

    def step(self, x, y, yaw, v, cmd, dt):
        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
        yaw += v * float(self.curvature(cmd)) * dt
        return x, y, yaw

    def rollout(self, x, y, yaw, v_seq, cmd_seq, dt_seq):
        """Returns an (N+1, 3) array of [x, y, yaw], starting with the initial state."""
        traj = [(x, y, yaw)]
        for v, cmd, dt in zip(v_seq, cmd_seq, dt_seq):
            x, y, yaw = self.step(x, y, yaw, v, cmd, dt)
            traj.append((x, y, yaw))
        return np.array(traj)

# ================= MPC Controller =================
class MPC_Controller:
    def __init__(self, horizon=20, dt_mpc=0.03, wheelbase=2.0, max_steer=np.radians(35)):
        self.horizon = horizon
        self.dt_mpc = dt_mpc        # MPC prediction timestep
        self.wheelbase = wheelbase
        self.max_steer = max_steer
        self.k_y = 1
        self.k_yaw = 0.75
        self.k_smooth = 0.25
        self.target_speed = 2.0 # 7.5

    def normalize_angle(self, angle):
        return (angle + np.pi) % (2*np.pi) - np.pi

    def predict_trajectory(self, x0, y0, yaw0, steering_sequence):
        x_pred, y_pred, yaw_pred = x0, y0, yaw0
        traj = []
        for steer in steering_sequence:
            yaw_pred += (self.target_speed / self.wheelbase) * math.tan(steer) * self.dt_mpc
            x_pred += self.target_speed * math.cos(yaw_pred) * self.dt_mpc
            y_pred += self.target_speed * math.sin(yaw_pred) * self.dt_mpc
            traj.append((x_pred, y_pred, yaw_pred))
        return traj

    def cost_function(self, steering_sequence, vehicle, path, closest_idx):
        steering_sequence = np.clip(steering_sequence, -self.max_steer, self.max_steer)
        traj = self.predict_trajectory(vehicle.x, vehicle.y, vehicle.heading, steering_sequence)
        cost = 0.0
        idx = closest_idx
        prev_steer = 0.0
        for i, (x_pred, y_pred, yaw_pred) in enumerate(traj):
            idx = min(idx + 1, len(path)-1)
            x_ref, y_ref, yaw_ref = path[idx]
            # dyaw = self.normalize_angle(yaw_ref - yaw_pred)
            #cost += self.k_y * ((x_ref - x_pred)**2 + (y_ref - y_pred)**2) + self.k_yaw * dyaw**2
            cost += self.k_y * ((x_ref - x_pred)**2 + (y_ref - y_pred)**2)  
            cost += self.k_smooth * (steering_sequence[i] - prev_steer)**2
            prev_steer = steering_sequence[i]
        return cost

    def run(self, vehicle, path, closest_idx):
        x0 = np.zeros(self.horizon)
        bounds = [(-self.max_steer, self.max_steer)] * self.horizon
        res = minimize(self.cost_function, x0, args=(vehicle, path, closest_idx),
                       bounds=bounds, method='SLSQP', options={'ftol':1e-3, 'disp':False, 'maxiter':30})
        best_sequence = res.x
        angular_velocity = (self.target_speed / self.wheelbase) * math.tan(best_sequence[0])
        return angular_velocity, self.target_speed

# ================= Utilities =================
def create_path_csv(filename):
    df = pd.read_csv(filename, sep=';', skiprows=2)
    df.columns = df.columns.str.strip()
    df.rename(columns={'x_m':'x','y_m':'y'}, inplace=True)
    return df[['x','y','psi_rad']].to_numpy()

def find_closest_point(path, x, y):
    distances = np.sqrt((path[:,0]-x)**2 + (path[:,1]-y)**2)
    return np.argmin(distances)

def get_initial_state(path):
    start_x = path[0,0]
    start_y = path[0,1]
    return start_x, start_y, 0
    n_segments = 5
    headings = [math.atan2(path[i+1,1]-path[i,1], path[i+1,0]-path[i,0]) for i in range(n_segments)]
    avg_heading = math.atan2(sum(math.sin(h) for h in headings),
                             sum(math.cos(h) for h in headings))
    cam_offset = 0.6096
    start_x = path[0,0] - cam_offset * math.cos(avg_heading)
    start_y = path[0,1] - cam_offset * math.sin(avg_heading)
    return start_x, start_y, avg_heading

# ================= Simulation =================
def simulate_mpc(csv_file):
    start = time.time()
    dt_sim = 0.01
    total_time = 200        # smaller for testing
    path = create_path_csv(csv_file)
    start_x, start_y, start_heading = get_initial_state(path)

    vehicle = KinematicBicycleModel(start_x, start_y, start_heading, 2.0, dt_sim)
    mpc_ctrl = MPC_Controller(horizon=5, dt_mpc=0.1)

    traj_x, traj_y = [vehicle.x], [vehicle.y]
    lap_done = False
    lap_time = None

    for t in np.arange(0, total_time, dt_sim):
        if lap_done:
            break
        idx = find_closest_point(path, vehicle.x, vehicle.y)
        ang_vel, speed = mpc_ctrl.run(vehicle, path, idx)
        
        vehicle.update(speed, ang_vel)
        traj_x.append(vehicle.x)
        traj_y.append(vehicle.y)

        if t > 5.0 and math.hypot(vehicle.x - path[0,0], vehicle.y - path[0,1]) < 1.0:
            lap_done = True
            lap_time = t

    end = time.time()
    print(end - start)
    plt.figure(figsize=(10,6))
    plt.plot(path[:,0], path[:,1], 'r--', label='Target Path')
    plt.plot(traj_x, traj_y, 'g-', label='MPC Trajectory')
    plt.scatter(traj_x[0], traj_y[0], color='green', marker='o', label='Start')
    plt.scatter(path[-1,0], path[-1,1], color='red', marker='x', label='End')
    if lap_done:
        plt.text(traj_x[-1], traj_y[-1], f"MPC Lap: {lap_time:.2f}s",
                 fontsize=12, color='green', ha='left', va='bottom',
                 bbox=dict(facecolor='white', edgecolor='green', boxstyle='round,pad=0.3'))
    else:
        plt.text(traj_x[-1], traj_y[-1], "MPC: Incomplete", fontsize=12)
    plt.title("MPC Controller with Coarse Prediction Step")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()



# ================= DonkeyCar Parts =================

class MPC_Part:
    """
    DonkeyCar part wrapper for MPC controller.
    Takes GPS position and yaw, outputs desired angular velocity.
    """
    def __init__(self, path_csv, horizon, dt_mpc, wheelbase, max_steer):
        
        self.path = self._load_path(path_csv)
        
        self.mpc = MPC_Controller(
            horizon=horizon,
            dt_mpc=dt_mpc,
            wheelbase=wheelbase,
            max_steer=max_steer
        )
        
        print(f"[MPC_Part] Initialized with {len(self.path)} waypoints from {path_csv}")
        
    def _load_path(self, filename):
        
        try:
            df = pd.read_csv(filename)
            df.columns = df.columns.str.strip()
            df.rename(columns={'x_m':'x', 'y_m':'y'}, inplace=True)
            path = df[['x', 'y', 'psi_rad']].to_numpy()
            return path
        except Exception as e:
            print(f"[MPC_Part] Error loading path from {filename}: {e}")
            return np.array([[0, 0, 0], [10, 0, 0]])
    
    def run(self, gps_x, gps_y, yaw):
        """
        Run MPC controller.
        Inputs: GPS position and yaw, outputs desired angular velocity and target speed.
        """

        if gps_x is None or gps_y is None or yaw is None:
            print("[MPC_Part] Waiting for valid GPS/yaw data...")
            return 0.0, 0.0
        
        class VehicleState:
            def __init__(self, x, y, heading):
                self.x = x
                self.y = y
                self.heading = heading
                self.wheelbase = 1.000506
                self.dt = 0.01
        
        vehicle = VehicleState(gps_x, gps_y, yaw)
        
        distances = np.sqrt((self.path[:, 0] - gps_x)**2 + 
                          (self.path[:, 1] - gps_y)**2)
        closest_idx = np.argmin(distances)

        # Print dx, dy to next point
        next_point = self.path[closest_idx]
        dx = next_point[0] - gps_x
        dy = next_point[1] - gps_y
        print(f"[MPC_Part] dx={dx:.2f}, dy={dy:.2f}, closest_idx={closest_idx}, idx_x = {next_point[0]:.2f}, idx_y = {next_point[1]:.2f}")
        
        # try:
        desired_ang_vel, target_speed = self.mpc.run(vehicle, self.path, closest_idx)
        print(f"MPC_Part:desired_ang_vel:{desired_ang_vel}, targert_speed:{target_speed}")
        return desired_ang_vel, target_speed
        # except Exception as e:
        #     print(f"[MPC_Part] Error in control loop: {e}")
        #     return 0.0, 0.0
    
    def shutdown(self):
        print("[MPC_Part] Shutting down")


class ClosedLoopController:
    """
    PID controller to convert desired angular velocity to steering.
    Takes desired angular velocity from MPC and actual angular velocity from IMU,
    from IMU, uses closed-loop feedback to output steering commands.
    """

    def __init__(self, kp, ki, kd, wheelbase, steering_scale, max_steering_deg):
        
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        self.wheelbase = wheelbase
        self.steering_scale = steering_scale
        self.max_steering_deg = max_steering_deg
        
        self.error_integral = 0.0
        self.last_error = 0.0
        self.last_time = None
        
        # upper limit for integral term to prevent integral from growing unbounded
        self.integral_limit = 10.0
        
        print(f"[PID] Initialized with kp={kp}, ki={ki}, kd={kd}")
        print(f"[PID] Steering: max={max_steering_deg}°, scale={steering_scale}")
        
    def run(self, desired_ang_vel, actual_ang_vel, target_speed):

        if desired_ang_vel is None or actual_ang_vel is None or target_speed is None:
            print("[PID] Waiting for valid inputs...")
            return 0.0, 0.0
        
        error = desired_ang_vel - actual_ang_vel
        
        current_time = time.time()
        if self.last_time is None:
            dt = 0.01 
        else:
            dt = current_time - self.last_time
            dt = max(dt, 0.001)  
        self.last_time = current_time
        
        p_term = self.kp * error
        self.error_integral += error * dt
        # prevent integral from growing unbounded
        self.error_integral = np.clip(self.error_integral, -self.integral_limit, self.integral_limit)
        i_term = self.ki * self.error_integral
        
        if dt > 0:
            d_term = self.kd * (error - self.last_error) / dt
        else:
            d_term = 0.0
        self.last_error = error
        

        # if abs(target_speed) > 0.1:
        #     feedforward_angle = math.atan((desired_ang_vel * self.wheelbase) / target_speed)
        # else:
        feedforward_angle = 0.0  
        
        feedback_correction = p_term + i_term + d_term
        total_steering_rad = feedforward_angle + feedback_correction
        
        # Convert to degrees
        total_steering_deg = total_steering_rad * (180.0 / math.pi)
        total_steering_deg = np.clip(total_steering_deg, -self.max_steering_deg, self.max_steering_deg)
        
        # Normalize
        steering_value = total_steering_deg * self.steering_scale
        steering_value = -1.0 * np.clip(steering_value, -1.0, 1.0) # Invert b.c angvel increments left, steer is right
        
        throttle = target_speed
        
        print(f"[PID] Desired: {desired_ang_vel:.3f} rad/s | "
              f"Actual: {actual_ang_vel:.3f} rad/s | "
              f"Error: {error:.3f} | "
              f"P: {p_term:.3f} I: {i_term:.3f} D: {d_term:.3f} | "
              f"FF: {feedforward_angle:.3f} rad | "
              f"Steer: {steering_value:.3f}")
        
        return steering_value, throttle
    
    def reset(self):
        self.error_integral = 0.0
        self.last_error = 0.0
        self.last_time = None
        print("[PID] State reset")
    
    def shutdown(self):
        print("[PID] Shutting down")


if __name__ == "__main__":
    simulate_mpc("traj_race_cl.csv")
