#!/usr/bin/env python3
"""ROS 2 port of parts/image_publisher.py.

publishes: camera/image_raw (Image)
"""

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from ros2_parts.parameters import declare


class ImagePublisherNode(Node):
    def __init__(self):
        super().__init__("image_publisher")

        device = declare(self, "device", 0, "OpenCV capture device index.")
        self.frame_id = declare(self, "frame_id", "camera", "Frame the images are taken in.")
        rate_hz = declare(self, "rate_hz", 30.0, "Rate frames are grabbed at, in Hz.")

        self.cap = cv2.VideoCapture(device)
        if not self.cap.isOpened():
            self.get_logger().error(f"Could not open capture device {device}")
        self.frame = None

        self.bridge = CvBridge()
        self.pub = self.create_publisher(Image, "camera/image_raw", qos_profile_sensor_data)
        self.create_timer(1.0 / rate_hz, self.publish_frame)

    def run(self):
        ret, self.frame = self.cap.read()
        if self.frame is not None:
            return self.frame

    def publish_frame(self):
        frame = self.run()
        if frame is None:
            self.get_logger().warn("no frame from the camera", throttle_duration_sec=5.0)
            return

        msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        self.pub.publish(msg)

    def destroy_node(self):
        if self.cap is not None:
            self.cap.release()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ImagePublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
