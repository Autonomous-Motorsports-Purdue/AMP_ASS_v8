#!/usr/bin/env python3
"""ROS 2 port of parts/gps_visualizer.py.

Plots the GPS track and heading live in matplotlib. Drawing happens on a
timer, which rclpy.spin runs on the main thread, where matplotlib's event
loop needs to be.

subscribes: gps/fix (NavSatFix), imu/data (Imu)
"""

import math
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, NavSatFix

from ros2_parts.orientation import quaternion_to_yaw
from ros2_parts.parameters import declare

BUFFER_SIZE = 500

# meters per degree of latitude, near enough for setting a view window
METERS_PER_DEGREE = 111_320.0


class GpsVisualizerNode(Node):
    def __init__(self):
        super().__init__("gps_visualizer")

        path_csv = declare(
            self, "path_csv", "", "Waypoint CSV drawn behind the track; empty draws none.")
        buffer_size = declare(self, "buffer_size", BUFFER_SIZE, "Fixes kept in the trail.")
        view_span_m = declare(self, "view_span_m", 80.0, "Width of the view window, in meters.")
        redraw_hz = declare(self, "redraw_hz", 15.0, "Rate the plot is redrawn at, in Hz.")

        self.lats = deque(maxlen=buffer_size)
        self.lons = deque(maxlen=buffer_size)
        self.view_span_deg = view_span_m / METERS_PER_DEGREE
        self.yaw = 0.0
        self.yaw_arrow = None

        if path_csv:
            data = np.genfromtxt(
                path_csv, delimiter=',', skip_header=1, dtype=float, encoding='utf-8')
            self.lat_path = data[:, 0]
            self.lon_path = data[:, 1]
        else:
            self.lat_path = None
            self.lon_path = None

        self.fig, self.ax = plt.subplots(figsize=(9, 9))
        self.ax.set_title("GPS Track")
        self.ax.set_xlabel("Longitude")
        self.ax.set_ylabel("Latitude")

        (self.trail_line,) = self.ax.plot(
            [], [], '-', color='#00ff88', linewidth=2, alpha=0.8, zorder=3)
        (self.current_dot,) = self.ax.plot(
            [], [], 'o', color='#ff3333', markersize=10, zorder=4)
        if self.lat_path is not None and self.lon_path is not None:
            (self.path,) = self.ax.plot(
                self.lon_path, self.lat_path, '--', color='#3333ff',
                linewidth=2, alpha=0.5, zorder=2)

        plt.ion()
        plt.show(block=False)

        self.create_subscription(NavSatFix, "gps/fix", self.on_fix, qos_profile_sensor_data)
        self.create_subscription(Imu, "imu/data", self.on_imu, qos_profile_sensor_data)
        self.create_timer(1.0 / redraw_hz, self.redraw)

    def on_fix(self, fix):
        self.lats.append(fix.latitude)
        self.lons.append(fix.longitude)

    def on_imu(self, imu):
        self.yaw = quaternion_to_yaw(imu.orientation)

    def _follow_view(self, lat, lon):
        """Recenter the axis around (lat, lon) at self.view_span_deg. A degree
        of longitude shrinks toward the poles, so keep the window square."""
        half_lat = 0.5 * self.view_span_deg
        half_lon = half_lat / max(math.cos(math.radians(lat)), 1e-3)
        self.ax.set_xlim(lon - half_lon, lon + half_lon)
        self.ax.set_ylim(lat - half_lat, lat + half_lat)

    def run(self, lat_deg, lon_deg, yaw):
        self._draw_sample(lat_deg, lon_deg, yaw)

    def redraw(self):
        if not self.lats:
            return
        self.run(self.lats[-1], self.lons[-1], self.yaw)

    def _draw_sample(self, lat_deg, lon_deg, yaw):
        lats = list(self.lats)
        lons = list(self.lons)

        self.trail_line.set_data(lons, lats)
        self.current_dot.set_data([lons[-1]], [lats[-1]])

        self._follow_view(lats[-1], lons[-1])

        theta = yaw
        lat_span = max(max(lats) - min(lats), 1e-5)
        lon_span = max(max(lons) - min(lons), 1e-5)
        arrow_len = 0.08 * max(lat_span, lon_span)

        dx = arrow_len * math.cos(theta)
        dy = arrow_len * math.sin(theta)

        if self.yaw_arrow is None:
            self.yaw_arrow = self.ax.annotate(
                "",
                xy=(lons[-1] + dx, lats[-1] + dy),
                xytext=(lons[-1], lats[-1]),
                arrowprops=dict(arrowstyle="->", color="#ffd400", lw=2),
                zorder=5,
            )
        else:
            self.yaw_arrow.xy = (lons[-1] + dx, lats[-1] + dy)
            self.yaw_arrow.set_position((lons[-1], lats[-1]))

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def destroy_node(self):
        plt.close(self.fig)
        return super().destroy_node()


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
