#!/usr/bin/env python3
# in: perception/waypoint (PointStamped, px) | out: perception/waypoint_ground (PointStamped, meters on the ground in base_link)
import pickle

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped


class TranslateNode(Node):
    def __init__(self):
        super().__init__("translate")
        calib_path = self.declare_parameter("calibration_path", "zed_calibration_params.bin").value
        self.height = self.declare_parameter("height", 0.6).value
        self.pitch = self.declare_parameter("pitch", 0.0).value

        with open(calib_path, "rb") as f:
            zed_calib = pickle.load(f)
        self.camera_matrix = np.array([[zed_calib["fx"], 0, zed_calib["cx"]],
                                       [0, zed_calib["fy"], zed_calib["cy"]],
                                       [0, 0, 1]])

        self.create_subscription(PointStamped, "perception/waypoint", self.run, 10)
        self.pub = self.create_publisher(PointStamped, "perception/waypoint_ground", 10)

    def run(self, msg):
        v_col, u_row = msg.point.x * 2, msg.point.y * 2  # u=y, v=x
        r = np.deg2rad(self.pitch)
        R_wc = np.array([[0, np.sin(r), np.cos(r)],
                         [-1, 0, 0],
                         [0, -np.cos(r), np.sin(r)]], dtype=np.float64)

        # camera optical centre in world coords
        C_w = np.array([0.0, 0.0, self.height], dtype=np.float64)

        # pixel -> ray in camera frame -> ray in world frame
        ray_cam = np.linalg.inv(self.camera_matrix) @ np.array([v_col, u_row, 1.0])
        ray_world = R_wc @ ray_cam

        # intersect with the ground plane (Z = 0)
        if abs(ray_world[2]) < 1e-9:
            return
        lam = -C_w[2] / ray_world[2]
        P_w = C_w + lam * ray_world

        out = PointStamped()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = "base_link"
        out.point.x = float(P_w[0])
        out.point.y = float(P_w[1])
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
