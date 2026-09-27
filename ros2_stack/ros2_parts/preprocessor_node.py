#!/usr/bin/env python3
# in: camera/image_raw (Image) | out: camera/image_preprocessed (Image)
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

SKY_RATIO = 330 / 720
CAR_RATIO = 163 / 720


class PreprocessorNode(Node):
    def __init__(self):
        super().__init__("preprocessor")
        self.bridge = CvBridge()
        self.create_subscription(Image, "camera/image_raw", self.run, qos_profile_sensor_data)
        self.pub = self.create_publisher(Image, "camera/image_preprocessed", qos_profile_sensor_data)

    def run(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        imgh, imgw = len(img), len(img[0])

        sky_crop = int(imgh * SKY_RATIO)
        car_crop = int(imgh * CAR_RATIO)

        img = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_AREA)

        brightness = np.sum(img, axis=-1)
        brightness = np.repeat(brightness[..., np.newaxis], 3, axis=-1)
        img = np.where(brightness < 100, 60, img).astype(np.uint8)

        cv2.rectangle(img, (0, 0), (imgw, sky_crop), (0, 0, 0), -1)
        cv2.rectangle(img, (0, imgh - car_crop), (imgw, imgh), (0, 0, 0), -1)

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
