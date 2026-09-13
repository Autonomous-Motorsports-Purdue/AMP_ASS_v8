#!/usr/bin/env python3
"""ROS 2 port of parts/imu_visualizer.py.

Plots the IMU outputs in matplotlib: yaw rate, yaw, acceleration and a
compass. Drawing happens on a timer, which rclpy.spin runs on the main
thread, where matplotlib's event loop needs to be.

subscribes: imu/data (Imu)
"""

import math
import time
from collections import deque

import matplotlib.pyplot as plt
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu

from ros2_parts.orientation import quaternion_to_yaw
from ros2_parts.parameters import declare

BUFFER_SIZE = 500


class ImuVisualizerNode(Node):
    def __init__(self):
        super().__init__("imu_visualizer")

        buffer_size = declare(self, "buffer_size", BUFFER_SIZE, "Samples kept per trace.")
        self.time_window_s = declare(
            self, "time_window_s", 20.0, "Width of the plotted window, in seconds.")
        redraw_hz = declare(self, "redraw_hz", 15.0, "Rate the plot is redrawn at, in Hz.")

        self.t = deque(maxlen=buffer_size)
        self.yaw_rates = deque(maxlen=buffer_size)
        self.yaws = deque(maxlen=buffer_size)
        self.axs = deque(maxlen=buffer_size)
        self.ays = deque(maxlen=buffer_size)

        self._t0 = time.time()
        self.latest = None

        self.fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        self.fig.suptitle("IMU")
        self.ax_yawrate, self.ax_yaw = axes[0]
        self.ax_accel, self.ax_compass = axes[1]

        self.ax_yawrate.set_title("Yaw rate (rad/s)")
        self.ax_yawrate.set_xlabel("t (s)")
        self.ax_yawrate.grid(True, alpha=0.3)
        (self.line_yawrate,) = self.ax_yawrate.plot([], [], color="#00ff88", lw=1.5)

        self.ax_yaw.set_title("Yaw (deg)")
        self.ax_yaw.set_xlabel("t (s)")
        self.ax_yaw.grid(True, alpha=0.3)
        (self.line_yaw,) = self.ax_yaw.plot([], [], color="#3399ff", lw=1.5)

        self.ax_accel.set_title("Acceleration (m/s^2)")
        self.ax_accel.set_xlabel("t (s)")
        self.ax_accel.grid(True, alpha=0.3)
        (self.line_ax,) = self.ax_accel.plot([], [], color="#ff5555", lw=1.5, label="ax")
        (self.line_ay,) = self.ax_accel.plot([], [], color="#ffaa00", lw=1.5, label="ay")
        self.ax_accel.legend(loc="upper right")

        self.ax_compass.set_title("Heading")
        self.ax_compass.set_aspect("equal")
        self.ax_compass.set_xlim(-1.2, 1.2)
        self.ax_compass.set_ylim(-1.2, 1.2)
        self.ax_compass.axhline(0, color="#888", lw=0.5)
        self.ax_compass.axvline(0, color="#888", lw=0.5)
        circle = plt.Circle((0, 0), 1.0, fill=False, color="#888")
        self.ax_compass.add_patch(circle)
        (self.heading_arrow,) = self.ax_compass.plot([0, 1], [0, 0], color="#ffd400", lw=2)

        self.fig.tight_layout()
        plt.ion()
        plt.show(block=False)

        self.create_subscription(Imu, "imu/data", self.on_imu, qos_profile_sensor_data)
        self.create_timer(1.0 / redraw_hz, self.redraw)

    def on_imu(self, imu):
        self.latest = imu

    def run(self, yaw_rate, yaw, ax, ay):
        if yaw_rate is None:
            return
        self._draw_sample(yaw_rate, yaw, ax, ay, time.time() - self._t0)

    def redraw(self):
        if self.latest is None:
            return
        self.run(
            self.latest.angular_velocity.z,
            quaternion_to_yaw(self.latest.orientation),
            self.latest.linear_acceleration.x,
            self.latest.linear_acceleration.y)

    def _draw_sample(self, yaw_rate, yaw, ax, ay, t):
        self.t.append(t)
        self.yaw_rates.append(float(yaw_rate))
        self.yaws.append(math.degrees(float(yaw)))
        self.axs.append(float(ax))
        self.ays.append(float(ay))

        ts = list(self.t)
        self.line_yawrate.set_data(ts, list(self.yaw_rates))
        self.line_yaw.set_data(ts, list(self.yaws))
        self.line_ax.set_data(ts, list(self.axs))
        self.line_ay.set_data(ts, list(self.ays))

        t_max = ts[-1]
        t_min = max(ts[0], t_max - self.time_window_s)
        for a in (self.ax_yawrate, self.ax_yaw, self.ax_accel):
            a.set_xlim(t_min, t_max if t_max > t_min else t_min + 1e-3)
            a.relim()
            a.autoscale_view(scalex=False, scaley=True)

        theta = float(yaw)
        self.heading_arrow.set_data([0, math.cos(theta)], [0, math.sin(theta)])

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def destroy_node(self):
        plt.close(self.fig)
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ImuVisualizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
