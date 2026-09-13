#!/usr/bin/env python3
"""ROS 2 port of parts/logger.py.

Saves camera and segmented frames to disk alongside a CSV of the controls.
The directory layout and CSV columns are unchanged, so existing tooling still
reads the output. A row is written on each segmented frame, pairing it with
the most recent camera frame and controls.

subscribes: camera/left/image_raw, perception/segmented_image (Image),
            perception/centroid (PointStamped),
            cmd/steering, cmd/throttle (Float64)
"""

import csv
import datetime
import os

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Image
from std_msgs.msg import Float64

from ros2_parts.parameters import declare


class LoggerNode(Node):
    def __init__(self):
        super().__init__("logger")

        data_dir = declare(self, "data_dir", "data", "Root directory for the recording.")

        start_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.image_directory = os.path.join(data_dir, "images", start_time)
        self.segmented_directory = os.path.join(data_dir, "segmented_images", start_time)
        self.information_directory = os.path.join(data_dir, "information")
        for directory in (self.image_directory, self.segmented_directory,
                          self.information_directory):
            if not os.path.exists(directory):
                os.makedirs(directory)

        self.info_csv = os.path.join(self.information_directory, start_time + ".csv")
        # Writing to csv file
        self.csvfile = open(self.info_csv, "w", newline="")
        # Creating a csv writer object
        self.csvwriter = csv.writer(self.csvfile)

        fields = ["timestamp", "image", "segmented", "centroid", "steering", "throttle"]
        # Writing the fields
        self.csvwriter.writerow(fields)
        self.get_logger().info(f"Recording to {data_dir}")

        self.bridge = CvBridge()
        self.image = None
        self.centroid = None
        self.steering = None
        self.throttle = None

        self.create_subscription(
            Image, "camera/left/image_raw", self.on_image, qos_profile_sensor_data)
        self.create_subscription(PointStamped, "perception/centroid", self.on_centroid, 10)
        self.create_subscription(Float64, "cmd/steering", self.on_steering, 10)
        self.create_subscription(Float64, "cmd/throttle", self.on_throttle, 10)
        self.create_subscription(
            Image, "perception/segmented_image", self.on_segmented, qos_profile_sensor_data)

    def on_image(self, msg):
        self.image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def on_centroid(self, msg):
        self.centroid = (msg.point.x, msg.point.y)

    def on_steering(self, msg):
        self.steering = msg.data

    def on_throttle(self, msg):
        self.throttle = msg.data

    def run(self, image, segmentedImage, centroid, steering, throttle):
        """
        Logs the current image, segmented Image, centroid, steering, and throttle values.
        Saves the images in their respective directory and logs the image paths and other data into a CSV.
        """
        if image is not None and segmentedImage is not None:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%f")
            image_file = self.image_directory + "/" + timestamp + ".jpg"
            segmented_file = self.segmented_directory + "/" + timestamp + ".jpg"

            # Save the images if written is successful
            success_image = cv2.imwrite(image_file, image.copy())
            success_segmented = cv2.imwrite(segmented_file, segmentedImage.copy())
            if success_image and success_segmented:
                rows = [timestamp, image_file, segmented_file, centroid, steering, throttle]
                self.csvwriter.writerow(rows)
                self.csvfile.flush()
            else:
                self.get_logger().warn("could not write frames", throttle_duration_sec=5.0)

    def on_segmented(self, msg):
        self.run(
            self.image,
            self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8"),
            self.centroid, self.steering, self.throttle)

    def destroy_node(self):
        if self.csvfile is not None and not self.csvfile.closed:
            self.csvfile.close()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LoggerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
