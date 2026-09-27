#!/usr/bin/env python3
# in: camera/left/image_raw, perception/segmented_image (Image), perception/centroid (PointStamped), cmd/steering, cmd/throttle (Float64) | out: data/ images + csv
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


class LoggerNode(Node):
    def __init__(self):
        super().__init__("logger")
        start_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.image_directory = "data/images/" + start_time
        self.segmented_directory = "data/segmented_images/" + start_time
        os.makedirs(self.image_directory, exist_ok=True)
        os.makedirs(self.segmented_directory, exist_ok=True)
        os.makedirs("data/information", exist_ok=True)

        self.csvfile = open("data/information/" + start_time + ".csv", "w", newline="")
        self.csvwriter = csv.writer(self.csvfile)
        self.csvwriter.writerow(["timestamp", "image", "segmented", "centroid", "steering", "throttle"])

        self.bridge = CvBridge()
        self.image = None
        self.centroid = None
        self.steering = None
        self.throttle = None

        self.create_subscription(Image, "camera/left/image_raw", self.on_image, qos_profile_sensor_data)
        self.create_subscription(PointStamped, "perception/centroid", self.on_centroid, 10)
        self.create_subscription(Float64, "cmd/steering", self.on_steering, 10)
        self.create_subscription(Float64, "cmd/throttle", self.on_throttle, 10)
        self.create_subscription(Image, "perception/segmented_image", self.run, qos_profile_sensor_data)

    def on_image(self, msg):
        self.image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def on_centroid(self, msg):
        self.centroid = (msg.point.x, msg.point.y)

    def on_steering(self, msg):
        self.steering = msg.data

    def on_throttle(self, msg):
        self.throttle = msg.data

    def run(self, msg):
        # one row per segmented frame, paired with the latest camera frame and controls
        if self.image is None:
            return
        segmented = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%f")
        image_file = self.image_directory + "/" + timestamp + ".jpg"
        segmented_file = self.segmented_directory + "/" + timestamp + ".jpg"

        if cv2.imwrite(image_file, self.image) and cv2.imwrite(segmented_file, segmented):
            self.csvwriter.writerow([timestamp, image_file, segmented_file,
                                     self.centroid, self.steering, self.throttle])
            self.csvfile.flush()

    def destroy_node(self):
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
