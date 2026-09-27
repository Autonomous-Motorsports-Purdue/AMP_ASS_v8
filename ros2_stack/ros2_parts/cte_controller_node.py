#!/usr/bin/env python3
# in: odometry/filtered (Odometry), path_csv (x, y[, throttle]) | out: cmd/steering, cmd/throttle, controller/cross_track_error (Float64), controller/waypoint_index (Int32)
import time

import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64, Int32

from ros2_parts.geometry import cross_track_error


class PIDController:
    # donkeycar/parts/transform.py
    def __init__(self, p=0, i=0, d=0):
        self.Kp = p
        self.Ki = i
        self.Kd = d
        self.prev_tm = time.time()
        self.prev_feedback = 0
        self.integral = 0

    def run(self, target_value, feedback):
        curr_tm = time.time()
        dt = curr_tm - self.prev_tm
        error = target_value - feedback

        self.integral += error * dt
        alpha = self.Kp * error + self.Ki * self.integral
        if dt > 0:
            # D on measurement, so it opposes changes in feedback
            alpha -= self.Kd * (feedback - self.prev_feedback) / dt

        self.prev_tm = curr_tm
        self.prev_feedback = feedback
        return alpha


class CteControllerNode(Node):
    def __init__(self):
        super().__init__("cte_controller")
        path_csv = self.declare_parameter("path_csv", "").value
        kp = self.declare_parameter("kp", 0.25).value
        ki = self.declare_parameter("ki", 0.0).value
        kd = self.declare_parameter("kd", 0.1).value
        self.look_ahead = self.declare_parameter("look_ahead", 3).value
        self.look_behind = self.declare_parameter("look_behind", 1).value
        self.throttle = self.declare_parameter("throttle", 2500).value

        self.pid = PIDController(p=kp, i=ki, d=kd)

        # path is x,y[,throttle]; a header row means there's no throttle column
        a = np.atleast_2d(np.genfromtxt(path_csv, delimiter=",", dtype=float, encoding="utf-8"))
        if a.shape[1] >= 3 and np.isfinite(a[0, 0]) and np.isfinite(a[0, 1]):
            self.path_xy = a[:, :2]
            self.pwm_table = a[:, -1].astype(int)
        else:
            a = np.genfromtxt(path_csv, delimiter=",", dtype=float, encoding="utf-8", skip_header=1)
            self.path_xy = a[:, :2]
            self.pwm_table = None

        self.create_subscription(Odometry, "odometry/filtered", self.run, 10)
        self.steer_pub = self.create_publisher(Float64, "cmd/steering", 10)
        self.throt_pub = self.create_publisher(Float64, "cmd/throttle", 10)
        self.cte_pub = self.create_publisher(Float64, "controller/cross_track_error", 10)
        self.idx_pub = self.create_publisher(Int32, "controller/waypoint_index", 10)

    def run(self, odom):
        x, y = odom.pose.pose.position.x, odom.pose.pose.position.y

        # segment from look_behind before the nearest waypoint to look_ahead after it
        n = len(self.path_xy)
        idx = int(np.argmin(np.hypot(self.path_xy[:, 0] - x, self.path_xy[:, 1] - y)))
        a = self.path_xy[(idx - self.look_behind) % n]
        b = self.path_xy[(idx + self.look_ahead) % n]
        cte = cross_track_error(a, b, (x, y))

        steer = float(np.clip(-self.pid.run(0.0, cte), -1, 1))
        throttle = int(self.pwm_table[idx]) if self.pwm_table is not None else int(self.throttle)

        self.steer_pub.publish(Float64(data=steer))
        self.throt_pub.publish(Float64(data=float(throttle)))
        self.cte_pub.publish(Float64(data=float(cte)))
        self.idx_pub.publish(Int32(data=idx))


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
