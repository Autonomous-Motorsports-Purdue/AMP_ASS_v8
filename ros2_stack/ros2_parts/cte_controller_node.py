#!/usr/bin/env python3
"""ROS 2 port of parts/cte_controller.py.

Steers to minimize cross-track error against a recorded path. The
nearest-segment search and the PID on CTE are unchanged. The donkeycar.la
vector helpers the part used (Line3D, Vec3, dist) are now geometry.py, which
reproduces their sign convention.

The part also precomputed a path-curvature feedforward but had it commented
out of the final steering command, so it is not carried over.

subscribes: odometry/filtered (Odometry, fused pose),
            gps/pose (PoseStamped, GPS course),
            gps/vel (TwistStamped, ground speed)
publishes:  cmd/steering (Float64, normalized -1 to 1),
            cmd/throttle (Float64, eRPM from the path file),
            controller/cross_track_error (Float64, meters),
            controller/waypoint_index (Int32)
"""

import logging
import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64, Int32

from ros2_parts.geometry import cross_track_error, distance as dist
from ros2_parts.orientation import quaternion_to_yaw
from ros2_parts.parameters import declare
from ros2_parts.predictive_model_node import predict_step


# donkeycar/parts/transform.py
class PIDController:
    """ Performs a PID computation and returns a control value.
        This is based on the elapsed time (dt) and the current value
        of the process variable
        (i.e. the thing we're measuring and trying to change).
        https://github.com/chrisspen/pid_controller/blob/master/pid_controller/pid.py
    """

    def __init__(self, p=0, i=0, d=0, debug=False):

        # initialize gains
        self.Kp = p
        self.Ki = i
        self.Kd = d

        # The value the controller is trying to get the system to achieve.
        self.target = 0

        # initialize delta t variables
        self.prev_tm = time.time()
        self.prev_feedback = 0
        self.error = None
        self.integral = 0

        # initialize the output
        self.alpha = 0

        self.debug = debug

    def run(self, target_value, feedback):
        curr_tm = time.time()

        self.target = target_value
        error = self.error = self.target - feedback

        # Calculate time differential.
        dt = curr_tm - self.prev_tm

        # Initialize output variable.
        curr_alpha = 0

        # Add proportional component.
        curr_alpha += self.Kp * error

        # Add integral component.
        self.integral += error * dt
        curr_alpha += self.Ki * self.integral

        # Add differential component (avoiding divide-by-zero).
        if dt > 0:
            # D term should be opposing
            # feed - prev instead of prev - feed
            curr_alpha -= self.Kd * ((feedback - self.prev_feedback) / float(dt))

        # Maintain memory for next loop.
        self.prev_tm = curr_tm
        self.prev_feedback = feedback

        # Update the output
        self.alpha = curr_alpha

        return curr_alpha


# donkeycar/parts/path.py
class CTE(object):

    def __init__(self, look_ahead=1, look_behind=1, num_pts=None):
        self.num_pts = num_pts
        self.look_ahead = look_ahead
        self.look_behind = look_behind

    #
    # Find the index of the path element with minimal distance to (x,y).
    # This prefers the first element with the minimum distance if there
    # are more then one.
    #
    def nearest_pt(self, path, x, y, from_pt=0, num_pts=None):
        from_pt = from_pt if from_pt is not None else 0
        num_pts = num_pts if num_pts is not None else len(path)
        num_pts = min(num_pts, len(path))
        if num_pts < 0:
            logging.error("num_pts must not be negative.")
            return None, None, None

        min_pt = None
        min_dist = None
        min_index = None
        for j in range(num_pts):
            i = (j + from_pt) % len(path)
            p = path[i]
            d = dist(p[0], p[1], x, y)
            if min_dist is None or d < min_dist:
                min_pt = p
                min_dist = d
                min_index = i
        return min_pt, min_index, min_dist

    def nearest_waypoints(self, path, x, y, look_ahead=1, look_behind=1,
                          from_pt=0, num_pts=None):
        """
        Get the path elements around the closest element to the given (x,y)
        :return: index of first point, nearest point and last point in nearest path segments
        """
        if path is None or len(path) < 2:
            logging.error("path is none; cannot calculate nearest points")
            return None, None, None

        if look_ahead < 0:
            logging.error("look_ahead must be a non-negative number")
            return None, None, None
        if look_behind < 0:
            logging.error("look_behind must be a non-negative number")
            return None, None, None
        if (look_ahead + look_behind) > len(path):
            logging.error("the path is not long enough to supply the waypoints")
            return None, None, None

        _pt, i, _distance = self.nearest_pt(path, x, y, from_pt, num_pts)

        # get  start of segment
        a = (i + len(path) - look_behind) % len(path)

        # get the end of the segment
        b = (i + look_ahead) % len(path)

        return a, i, b

    def nearest_track(self, path, x, y, look_ahead=1, look_behind=1,
                      from_pt=0, num_pts=None):
        """
        Get the line segment around the closest point to the given (x,y)
        :return: start and end points of the nearest track and index of nearest point
        """
        a, i, b = self.nearest_waypoints(
            path, x, y, look_ahead, look_behind, from_pt, num_pts)

        return (path[a], path[b], i) if a is not None and b is not None else (None, None, None)

    def run(self, path, x, y, look_ahead, look_behind, from_pt=None):
        """
        Run cross track error algorithm
        :return: cross-track-error, index of nearest point, and segment endpoints
        """
        cte = 0.
        i = from_pt

        a, b, i = self.nearest_track(path, x, y,
                                     look_ahead=look_ahead, look_behind=look_behind,
                                     from_pt=from_pt, num_pts=None)

        if type(a) == np.ndarray and type(b) == np.ndarray:
            cte = cross_track_error(a, b, (x, y))
        else:
            logging.info(f"no nearest point to ({x},{y}))")
        return cte, i, a, b


class CteControllerNode(Node):
    def __init__(self):
        super().__init__("cte_controller")

        path_csv = declare(
            self, "path_csv", "", "Waypoint CSV in local XY, optionally with a PWM column.")
        kp = declare(self, "kp", 0.25, "Proportional gain on cross-track error.")
        ki = declare(self, "ki", 0.0, "Integral gain on cross-track error.")
        kd = declare(self, "kd", 0.1, "Derivative gain on cross-track error.")
        self.lookahead = declare(
            self, "look_ahead", 3, "Waypoints ahead of the nearest used for the segment.")
        self.lookbehind = declare(
            self, "look_behind", 1, "Waypoints behind the nearest used for the segment.")
        self.throttle = declare(
            self, "throttle", 2500, "Throttle used when the path has no PWM column.")
        self.pred_steps = declare(
            self, "prediction_steps", 0, "Steps of dead reckoning applied before the search.")
        self.pred_dt = declare(
            self, "prediction_step_dt", 0.02, "Seconds per prediction step.")

        self.cte = CTE(look_ahead=self.lookahead, look_behind=self.lookbehind)
        self.pid = PIDController(p=kp, i=ki, d=kd, debug=False)

        a = np.genfromtxt(path_csv, delimiter=',', dtype=float, encoding='utf-8', skip_header=0)
        if a.ndim == 1:
            a = np.reshape(a, (1, -1))
        if a.shape[1] >= 3 and np.isfinite(a[0, 0]) and np.isfinite(a[0, 1]):
            self.path_xy = a[:, :2]
            self.pwm_table = np.asarray(a[:, -1], dtype=int)
        else:
            a = np.genfromtxt(
                path_csv, delimiter=',', dtype=float, encoding='utf-8', skip_header=1)
            self.path_xy = a[:, :2]
            self.pwm_table = None
        self.prev_cte = None
        self.get_logger().info(f"Loaded {len(self.path_xy)} waypoints from {path_csv}")

        self.gps_speed = None
        self.gps_yaw_math_deg = None

        self.steer_pub = self.create_publisher(Float64, "cmd/steering", 10)
        self.throt_pub = self.create_publisher(Float64, "cmd/throttle", 10)
        self.cte_pub = self.create_publisher(Float64, "controller/cross_track_error", 10)
        self.idx_pub = self.create_publisher(Int32, "controller/waypoint_index", 10)

        self.create_subscription(PoseStamped, "gps/pose", self.on_gps_pose, 10)
        self.create_subscription(
            TwistStamped, "gps/vel", self.on_gps_velocity, qos_profile_sensor_data)
        self.create_subscription(Odometry, "odometry/filtered", self.on_odometry, 10)

    def on_gps_pose(self, pose):
        self.gps_yaw_math_deg = math.degrees(quaternion_to_yaw(pose.pose.orientation))

    def on_gps_velocity(self, vel):
        self.gps_speed = math.hypot(vel.twist.linear.x, vel.twist.linear.y)

    def run(self, x, y, gps_speed, gps_yaw_math_deg):
        # One/N step pseudo MPC - predict X/Y N steps into the future
        N = self.pred_steps
        x_future, y_future = x, y
        if N and gps_yaw_math_deg is not None:
            x_future, y_future = predict_step(
                x_future, y_future, gps_speed, math.radians(gps_yaw_math_deg),
                N * self.pred_dt
            )

        cte, idx, a, b = self.cte.run(
            self.path_xy, x_future, y_future,
            look_ahead=self.lookahead, look_behind=self.lookbehind)
        self.prev_cte = cte
        cte_steer = self.pid.run(0.0, cte) # we desire 0 cte

        cte_steer *= -1

        steer = cte_steer
        steer = np.clip(steer, -1, 1)

        if self.pwm_table is not None:
            i = 0 if idx is None else int(idx) % len(self.pwm_table)
            throttle = int(self.pwm_table[i])
        else:
            throttle = int(self.throttle)

        return throttle, steer, cte, idx

    def on_odometry(self, odom):
        throttle, steer, cte, idx = self.run(
            odom.pose.pose.position.x,
            odom.pose.pose.position.y,
            self.gps_speed,
            self.gps_yaw_math_deg)

        self.get_logger().debug(f"CTE: {cte:.4f} at idx {idx}, steer {steer:.3f}")

        self.steer_pub.publish(Float64(data=float(steer)))
        self.throt_pub.publish(Float64(data=float(throttle)))
        self.cte_pub.publish(Float64(data=float(cte)))
        self.idx_pub.publish(Int32(data=int(idx or 0)))


def main(args=None):
    rclpy.init(args=args)
    node = CteControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
