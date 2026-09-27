#!/usr/bin/env python3
# in: imu/data (Imu) | out: live matplotlib plots of yaw rate, yaw, accel and a compass
import math
import time
from collections import deque

import matplotlib.pyplot as plt
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu

from ros2_parts.orientation import quaternion_to_yaw

TIME_WINDOW_S = 20.0


class ImuVisualizerNode(Node):
    def __init__(self):
        super().__init__("imu_visualizer")
        self.t = deque(maxlen=500)
        self.yaw_rates = deque(maxlen=500)
        self.yaws = deque(maxlen=500)
        self.axs = deque(maxlen=500)
        self.ays = deque(maxlen=500)
        self.t0 = time.time()
        self.latest = None

        self.fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        (self.ax_yawrate, self.ax_yaw), (self.ax_accel, self.ax_compass) = axes
        self.ax_yawrate.set_title("Yaw rate (rad/s)")
        self.ax_yaw.set_title("Yaw (deg)")
        self.ax_accel.set_title("Acceleration (m/s^2)")
        (self.line_yawrate,) = self.ax_yawrate.plot([], [], color="#00ff88")
        (self.line_yaw,) = self.ax_yaw.plot([], [], color="#3399ff")
        (self.line_ax,) = self.ax_accel.plot([], [], color="#ff5555", label="ax")
        (self.line_ay,) = self.ax_accel.plot([], [], color="#ffaa00", label="ay")
        self.ax_accel.legend(loc="upper right")

        self.ax_compass.set_title("Heading")
        self.ax_compass.set_aspect("equal")
        self.ax_compass.set_xlim(-1.2, 1.2)
        self.ax_compass.set_ylim(-1.2, 1.2)
        self.ax_compass.add_patch(plt.Circle((0, 0), 1.0, fill=False, color="#888"))
        (self.heading_arrow,) = self.ax_compass.plot([0, 1], [0, 0], color="#ffd400", lw=2)

        self.fig.tight_layout()
        plt.ion()
        plt.show(block=False)

        self.create_subscription(Imu, "imu/data", self.on_imu, qos_profile_sensor_data)
        # matplotlib has to draw from the main thread, which is where rclpy runs timers
        self.create_timer(1.0 / 15.0, self.run)

    def on_imu(self, imu):
        self.latest = imu

    def run(self):
        if self.latest is None:
            return
        yaw = quaternion_to_yaw(self.latest.orientation)
        self.t.append(time.time() - self.t0)
        self.yaw_rates.append(self.latest.angular_velocity.z)
        self.yaws.append(math.degrees(yaw))
        self.axs.append(self.latest.linear_acceleration.x)
        self.ays.append(self.latest.linear_acceleration.y)

        ts = list(self.t)
        self.line_yawrate.set_data(ts, list(self.yaw_rates))
        self.line_yaw.set_data(ts, list(self.yaws))
        self.line_ax.set_data(ts, list(self.axs))
        self.line_ay.set_data(ts, list(self.ays))
        for a in (self.ax_yawrate, self.ax_yaw, self.ax_accel):
            a.set_xlim(max(ts[0], ts[-1] - TIME_WINDOW_S), ts[-1] + 1e-3)
            a.relim()
            a.autoscale_view(scalex=False, scaley=True)
        self.heading_arrow.set_data([0, math.cos(yaw)], [0, math.sin(yaw)])

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()


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
