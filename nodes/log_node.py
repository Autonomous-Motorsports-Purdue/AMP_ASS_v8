#!/usr/bin/env python3
# in: perception/object_centroid (PointStamped), perception/contour_area (Float64) | out: logger.csv
import csv
import datetime

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float64


class LogNode(Node):
    def __init__(self):
        super().__init__("log")
        path = self.declare_parameter("path", "logger.csv").value

        self.csvfile = open(path, "w", newline="")
        self.csvwriter = csv.writer(self.csvfile)
        self.csvwriter.writerow(["timestamp", "object_x", "object_y", "contour_area"])

        self.contour_area = None
        self.create_subscription(Float64, "perception/contour_area", self.on_area, 10)
        self.create_subscription(PointStamped, "perception/object_centroid", self.run, 10)

    def on_area(self, msg):
        self.contour_area = msg.data

    def run(self, msg):
        if self.contour_area is None:
            return
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%f")
        self.csvwriter.writerow([timestamp, msg.point.x, msg.point.y, self.contour_area])
        self.csvfile.flush()

    def destroy_node(self):
        self.csvfile.close()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LogNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
