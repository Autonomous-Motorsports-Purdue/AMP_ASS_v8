#!/usr/bin/env python3
"""ROS 2 port of parts/log.py.

A row is written on each centroid, using the most recent area.

subscribes: perception/object_centroid (PointStamped),
            perception/contour_area (Float64)
"""

import csv
import datetime

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float64

from ros2_parts.parameters import declare


class LogNode(Node):
    def __init__(self):
        super().__init__("log")

        path = declare(self, "path", "logger.csv", "CSV file the detections go to.")

        # Writing to a csv file
        self.csvfile = open(path, "w", newline="")
        # Creating a csv writer object
        self.csvwriter = csv.writer(self.csvfile)

        fields = ["timestamp", "object_x", "object_y", "contour_area"]
        # Write fields
        self.csvwriter.writerow(fields)
        self.get_logger().info(f"Logging detections to {path}")

        self.contour_area = None

        self.create_subscription(Float64, "perception/contour_area", self.on_area, 10)
        self.create_subscription(
            PointStamped, "perception/object_centroid", self.on_centroid, 10)

    def on_area(self, msg):
        self.contour_area = msg.data

    def run(self, object_x, object_y, contour_area):
        if object_x is not None and object_y is not None and contour_area is not None:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%f")
            row = [timestamp, object_x, object_y, contour_area]
            self.csvwriter.writerow(row)
            self.csvfile.flush()

    def on_centroid(self, msg):
        self.run(msg.point.x, msg.point.y, self.contour_area)

    def destroy_node(self):
        if self.csvfile is not None and not self.csvfile.closed:
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
