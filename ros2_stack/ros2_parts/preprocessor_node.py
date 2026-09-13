#!/usr/bin/env python3
"""ROS 2 port of parts/preprocessor.py.

subscribes: camera/image_raw (Image)
publishes:  camera/image_preprocessed (Image)
"""

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from ros2_parts.parameters import declare


class PreprocessorNode(Node):

    SKY_RATIO = 330 / 720
    CAR_RATIO = 163 / 720

    def __init__(self):
        super().__init__("preprocessor")

        self.width = declare(self, "width", 1280, "Width the frame is resized to.")
        self.height = declare(self, "height", 720, "Height the frame is resized to.")

        self.bridge = CvBridge()
        self.pub = self.create_publisher(
            Image, "camera/image_preprocessed", qos_profile_sensor_data)
        self.create_subscription(
            Image, "camera/image_raw", self.on_image, qos_profile_sensor_data)

    def run(self, img):
        imgh, imgw = len(img), len(img[0])

        sky_crop = int(imgh * PreprocessorNode.SKY_RATIO)
        car_crop = int(imgh * PreprocessorNode.CAR_RATIO)

        img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_AREA)

        brightness = np.sum(img, axis=-1)
        brightness = np.repeat(brightness[..., np.newaxis], 3, axis=-1)
        img = np.where(brightness < 100, 60, img).astype(np.uint8)

        cv2.rectangle(img, (0, 0), (imgw, sky_crop), (0, 0, 0), -1)
        cv2.rectangle(img, (0, imgh - car_crop), (imgw, imgh), (0, 0, 0), -1)

        return img

    def on_image(self, msg):
        img = self.run(self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8"))
        out = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
        out.header = msg.header
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = PreprocessorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
