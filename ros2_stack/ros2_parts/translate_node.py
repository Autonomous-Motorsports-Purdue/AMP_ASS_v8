#!/usr/bin/env python3
"""ROS 2 port of parts/translate.py.

Projects an image-plane waypoint onto the ground.

subscribes: perception/waypoint (PointStamped, pixels)
publishes:  perception/waypoint_ground (PointStamped, meters in base_link)
"""

import pickle

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped

from ros2_parts.parameters import declare


class TranslateNode(Node):
    def __init__(self):
        super().__init__("translate")

        calib_path = declare(
            self, "calibration_path", "zed_calibration_params.bin",
            "Pickle of the ZED intrinsics written by the ZED frame publisher.")
        self.height = declare(self, "height", 0.6, "Camera height above the ground, in meters.")
        self.pitch = declare(self, "pitch", 0.0, "Camera pitch below horizontal, in degrees.")
        self.frame_id = declare(
            self, "frame_id", "base_link", "Frame the ground point is expressed in.")

        with open(calib_path, "rb") as f:
            zed_calib = pickle.load(f)

        self.camera_matrix = np.array([[zed_calib["fx"], 0, zed_calib["cx"]],
                                [0, zed_calib["fy"], zed_calib["cy"]],
                                [0, 0, 1]])

        self.pub = self.create_publisher(PointStamped, "perception/waypoint_ground", 10)
        self.create_subscription(PointStamped, "perception/waypoint", self.on_waypoint, 10)

    def run(self, point):
        v_col, u_row = point # u=y, v=x
        v_col *= 2
        u_row *= 2
        r = np.deg2rad(self.pitch)
        R_wc = np.array([[0, np.sin(r), np.cos(r)],
            [-1, 0, 0],
            [0, -np.cos(r), np.sin(r)]], dtype=np.float64)

        # --- camera optical centre in world coords ---
        C_w = np.array([0.0, 0.0, self.height], dtype=np.float64)

        # Pixel -> ray in camera frame
        ray_cam = np.linalg.inv(self.camera_matrix) @ np.array([v_col, u_row, 1.0])

        # Ray -> world frame
        ray_world = R_wc @ ray_cam

        # Intersect with ground plane (Z = 0)
        if abs(ray_world[2]) < 1e-9:
            raise ValueError("Ray is parallel to the ground plane.")
        lam = -C_w[2] / ray_world[2]
        P_w = C_w + lam * ray_world
        P_w[2] = 0.0                    # enforce exact planarity
        return P_w

    def on_waypoint(self, msg):
        try:
            P_w = self.run((msg.point.x, msg.point.y))
        except ValueError as e:
            self.get_logger().warn(f"{e}", throttle_duration_sec=5.0)
            return

        out = PointStamped()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.frame_id
        out.point.x = float(P_w[0])
        out.point.y = float(P_w[1])
        out.point.z = 0.0
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = TranslateNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
