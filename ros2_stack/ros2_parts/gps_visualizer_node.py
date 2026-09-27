#!/usr/bin/env python3
# in: gps/fix (NavSatFix), imu/data (Imu), optional path_csv | out: live matplotlib plot of the GPS track and heading
import math
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, NavSatFix

from ros2_parts.orientation import quaternion_to_yaw

VIEW_SPAN_DEG = 80.0 / 111_320.0  # ~80 m window


class GpsVisualizerNode(Node):
    def __init__(self):
        super().__init__("gps_visualizer")
        path_csv = self.declare_parameter("path_csv", "").value

        self.lats = deque(maxlen=500)
        self.lons = deque(maxlen=500)
        self.yaw = 0.0

        self.fig, self.ax = plt.subplots(figsize=(9, 9))
        self.ax.set_title("GPS Track")
        self.ax.set_xlabel("Longitude")
        self.ax.set_ylabel("Latitude")
        if path_csv:
            data = np.genfromtxt(path_csv, delimiter=",", skip_header=1, dtype=float, encoding="utf-8")
            self.ax.plot(data[:, 1], data[:, 0], "--", color="#3333ff", linewidth=2, alpha=0.5)
        (self.trail_line,) = self.ax.plot([], [], "-", color="#00ff88", linewidth=2, alpha=0.8)
        (self.current_dot,) = self.ax.plot([], [], "o", color="#ff3333", markersize=10)
        (self.heading_line,) = self.ax.plot([], [], "-", color="#ffd400", linewidth=2)
        plt.ion()
        plt.show(block=False)

        self.create_subscription(NavSatFix, "gps/fix", self.on_fix, qos_profile_sensor_data)
        self.create_subscription(Imu, "imu/data", self.on_imu, qos_profile_sensor_data)
        # matplotlib has to draw from the main thread, which is where rclpy runs timers
        self.create_timer(1.0 / 15.0, self.run)

    def on_fix(self, fix):
        self.lats.append(fix.latitude)
        self.lons.append(fix.longitude)

    def on_imu(self, imu):
        self.yaw = quaternion_to_yaw(imu.orientation)

    def run(self):
        if not self.lats:
            return
        lat, lon = self.lats[-1], self.lons[-1]
        self.trail_line.set_data(list(self.lons), list(self.lats))
        self.current_dot.set_data([lon], [lat])

        # heading tick, 10% of the view window long
        arrow_len = 0.1 * VIEW_SPAN_DEG
        self.heading_line.set_data([lon, lon + arrow_len * math.cos(self.yaw)],
                                   [lat, lat + arrow_len * math.sin(self.yaw)])

        # follow the kart; a degree of longitude shrinks with latitude
        half_lat = 0.5 * VIEW_SPAN_DEG
        half_lon = half_lat / math.cos(math.radians(lat))
        self.ax.set_xlim(lon - half_lon, lon + half_lon)
        self.ax.set_ylim(lat - half_lat, lat + half_lat)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()


def main(args=None):
    rclpy.init(args=args)
    node = GpsVisualizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
