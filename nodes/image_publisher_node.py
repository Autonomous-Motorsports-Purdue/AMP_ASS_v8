#!/usr/bin/env python3
# in: OpenCV camera | out: camera/image_raw (Image)
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


class ImagePublisherNode(Node):
    def __init__(self):
        super().__init__("image_publisher")
        device = self.declare_parameter("device", 0).value
        rate_hz = self.declare_parameter("rate_hz", 30.0).value

        self.cap = cv2.VideoCapture(device)
        self.bridge = CvBridge()
        self.pub = self.create_publisher(Image, "camera/image_raw", qos_profile_sensor_data)
        self.create_timer(1.0 / rate_hz, self.run)

    def run(self):
        ret, frame = self.cap.read()
        if frame is None:
            return
        msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera"
        self.pub.publish(msg)

    def destroy_node(self):
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
