#!/usr/bin/env python3
"""ROS 2 port of parts/controller.py.

MPC path tracking and the closed-loop yaw-rate controller behind it. The
matplotlib simulation in the part is not carried over.

Two controllers live in this file, so it provides two executables:

mpc_part
    subscribes: gps/pose (PoseStamped)
    publishes:  cmd/desired_yaw_rate, cmd/desired_speed (Float64)

closed_loop_controller
    subscribes: cmd/desired_yaw_rate, cmd/desired_speed (Float64),
                imu/data (Imu)
    publishes:  cmd/steering, cmd/throttle (Float64)

Both yaw rates are now in rad/s. sensor_msgs/Imu specifies that unit, whereas
the donkeycar IMU part passed deg/s into the same subtraction the MPC's
radians went into. The gains need rechecking.
"""

import math
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64

from ros2_parts.orientation import quaternion_to_yaw
from ros2_parts.parameters import declare

# used when the path CSV cannot be read: ten meters straight ahead
FALLBACK_PATH = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])


class VehicleState:
    def __init__(self, x, y, heading):
        self.x = x
        self.y = y
        self.heading = heading


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
            cost += self.k_y * ((x_ref - x_pred)**2 + (y_ref - y_pred)**2)
            cost += self.k_smooth * (steering_sequence[i] - prev_steer)**2
            prev_steer = steering_sequence[i]
        return cost

    def run(self, vehicle, path, closest_idx, maxiter=30):
        x0 = np.zeros(self.horizon)
        bounds = [(-self.max_steer, self.max_steer)] * self.horizon
        res = minimize(self.cost_function, x0, args=(vehicle, path, closest_idx),
                       bounds=bounds, method='SLSQP',
                       options={'ftol':1e-3, 'disp':False, 'maxiter':maxiter})
        best_sequence = res.x
        angular_velocity = (self.target_speed / self.wheelbase) * math.tan(best_sequence[0])
        return angular_velocity, self.target_speed


class MpcPartNode(Node):
    """
    Takes GPS position and yaw, outputs desired angular velocity.

    Each pose triggers a fresh solve over horizon steering angles. Only the
    first is used and the rest are re-solved next time, which is what makes it
    receding-horizon rather than open-loop.
    """

    def __init__(self):
        super().__init__("mpc_part")

        path_csv = declare(self, "path_csv", "", "Waypoint CSV with x, y and psi_rad columns.")
        horizon = declare(self, "horizon", 2, "Steps the optimizer looks ahead.")
        dt_mpc = declare(self, "dt_mpc", 0.1, "Prediction timestep, in seconds.")
        wheelbase = declare(self, "wheelbase", 1.000506, "Vehicle wheelbase, in meters.")
        max_steer_deg = declare(
            self, "max_steer_deg", 35.0, "Steering angle at full lock, in degrees.")
        target_speed = declare(
            self, "target_speed", 2.0, "Speed the prediction assumes, in m/s.")
        self.maxiter = declare(
            self, "max_iterations", 30, "Optimizer iteration cap, to bound the solve time.")

        self.path = self._load_path(path_csv)

        self.mpc = MPC_Controller(
            horizon=horizon,
            dt_mpc=dt_mpc,
            wheelbase=wheelbase,
            max_steer=math.radians(max_steer_deg)
        )
        self.mpc.target_speed = target_speed

        self.get_logger().info(f"Initialized with {len(self.path)} waypoints from {path_csv}")

        self.yaw_rate_pub = self.create_publisher(Float64, "cmd/desired_yaw_rate", 10)
        self.speed_pub = self.create_publisher(Float64, "cmd/desired_speed", 10)
        self.create_subscription(PoseStamped, "gps/pose", self.on_pose, 10)

    def _load_path(self, filename):
        try:
            df = pd.read_csv(filename)
            df.columns = df.columns.str.strip()
            df.rename(columns={'x_m':'x', 'y_m':'y'}, inplace=True)
            path = df[['x', 'y', 'psi_rad']].to_numpy()
            return path
        except Exception as e:
            self.get_logger().error(
                f"Error loading path from {filename}: {e}, driving straight")
            return FALLBACK_PATH

    def run(self, gps_x, gps_y, yaw):
        """
        Inputs: GPS position and yaw. Outputs desired angular velocity and
        target speed.
        """
        vehicle = VehicleState(gps_x, gps_y, yaw)

        distances = np.sqrt((self.path[:, 0] - gps_x)**2 +
                          (self.path[:, 1] - gps_y)**2)
        closest_idx = np.argmin(distances)

        desired_ang_vel, target_speed = self.mpc.run(
            vehicle, self.path, closest_idx, self.maxiter)
        self.get_logger().debug(
            f"closest_idx={closest_idx}, desired_ang_vel={desired_ang_vel:.3f}, "
            f"target_speed={target_speed:.2f}")
        return desired_ang_vel, target_speed

    def on_pose(self, pose):
        desired_ang_vel, target_speed = self.run(
            pose.pose.position.x,
            pose.pose.position.y,
            quaternion_to_yaw(pose.pose.orientation))

        self.yaw_rate_pub.publish(Float64(data=float(desired_ang_vel)))
        self.speed_pub.publish(Float64(data=float(target_speed)))


class ClosedLoopControllerNode(Node):
    """
    PID controller to convert desired angular velocity to steering.
    Takes desired angular velocity from the MPC and actual angular velocity
    from the IMU, uses closed-loop feedback to output steering commands.
    """

    def __init__(self):
        super().__init__("closed_loop_controller")

        self.kp = declare(self, "kp", 0.2, "Proportional gain on yaw-rate error.")
        self.ki = declare(self, "ki", 0.0, "Integral gain on yaw-rate error.")
        self.kd = declare(self, "kd", 0.0, "Derivative gain on yaw-rate error.")
        self.max_steering_deg = declare(
            self, "max_steering_deg", 35.0, "Steering angle at full lock, in degrees.")
        self.steering_scale = declare(
            self, "steering_scale", 1 / 35.0, "Degrees to normalized-command scale.")
        # upper limit for integral term to prevent integral from growing unbounded
        self.integral_limit = declare(
            self, "integral_limit", 10.0, "Cap on the integral term.")
        rate_hz = declare(self, "rate_hz", 30.0, "Rate commands are published at, in Hz.")

        self.error_integral = 0.0
        self.last_error = 0.0
        self.last_time = None

        self.desired_ang_vel = None
        self.actual_ang_vel = None
        self.target_speed = None

        self.steer_pub = self.create_publisher(Float64, "cmd/steering", 10)
        self.throt_pub = self.create_publisher(Float64, "cmd/throttle", 10)
        self.create_subscription(Float64, "cmd/desired_yaw_rate", self.on_desired_yaw_rate, 10)
        self.create_subscription(Float64, "cmd/desired_speed", self.on_desired_speed, 10)
        self.create_subscription(Imu, "imu/data", self.on_imu, qos_profile_sensor_data)
        self.create_timer(1.0 / rate_hz, self.publish_command)

    def on_desired_yaw_rate(self, msg):
        self.desired_ang_vel = msg.data

    def on_desired_speed(self, msg):
        self.target_speed = msg.data

    def on_imu(self, msg):
        self.actual_ang_vel = msg.angular_velocity.z

    def run(self, desired_ang_vel, actual_ang_vel, target_speed):
        if desired_ang_vel is None or actual_ang_vel is None or target_speed is None:
            self.get_logger().debug("Waiting for valid inputs...")
            return None, None

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
        self.error_integral = np.clip(
            self.error_integral, -self.integral_limit, self.integral_limit)
        i_term = self.ki * self.error_integral

        d_term = self.kd * (error - self.last_error) / dt
        self.last_error = error

        feedforward_angle = 0.0

        feedback_correction = p_term + i_term + d_term
        total_steering_rad = feedforward_angle + feedback_correction

        # Convert to degrees
        total_steering_deg = total_steering_rad * (180.0 / math.pi)
        total_steering_deg = np.clip(
            total_steering_deg, -self.max_steering_deg, self.max_steering_deg)

        # Normalize
        steering_value = total_steering_deg * self.steering_scale
        steering_value = -1.0 * np.clip(steering_value, -1.0, 1.0) # Invert b.c angvel increments left, steer is right

        throttle = target_speed

        self.get_logger().debug(
            f"Desired: {desired_ang_vel:.3f} rad/s | "
            f"Actual: {actual_ang_vel:.3f} rad/s | "
            f"Error: {error:.3f} | "
            f"P: {p_term:.3f} I: {i_term:.3f} D: {d_term:.3f} | "
            f"Steer: {steering_value:.3f}")

        return steering_value, throttle

    def reset(self):
        self.error_integral = 0.0
        self.last_error = 0.0
        self.last_time = None

    def publish_command(self):
        steering_value, throttle = self.run(
            self.desired_ang_vel, self.actual_ang_vel, self.target_speed)
        if steering_value is None:
            return
        self.steer_pub.publish(Float64(data=float(steering_value)))
        self.throt_pub.publish(Float64(data=float(throttle)))


def main(args=None):
    rclpy.init(args=args)
    node = MpcPartNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


def main_closed_loop(args=None):
    rclpy.init(args=args)
    node = ClosedLoopControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
