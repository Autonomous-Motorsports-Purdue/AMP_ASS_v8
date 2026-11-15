#!/usr/bin/env python3
# in: ZED as a wide UVC camera | out: camera/left/image_raw, camera/right/image_raw (Image)
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


class FramePublisherNode(Node):
    def __init__(self):
        super().__init__("frame_publisher")
        device = self.declare_parameter("device", 0).value
        width = self.declare_parameter("width", 2560).value
        height = self.declare_parameter("height", 720).value
        rate_hz = self.declare_parameter("rate_hz", 30.0).value

        self.cap = cv2.VideoCapture(device)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.bridge = CvBridge()
        self.left_pub = self.create_publisher(Image, "camera/left/image_raw", qos_profile_sensor_data)
        self.right_pub = self.create_publisher(Image, "camera/right/image_raw", qos_profile_sensor_data)
        self.create_timer(1.0 / rate_hz, self.run)

    def run(self):
        ret, frame = self.cap.read()
        if frame is None:
            return

        # split the side-by-side frame into left and right eyes
        width = frame.shape[1]
        stamp = self.get_clock().now().to_msg()
        for pub, half in ((self.left_pub, frame[:, :width // 2]), (self.right_pub, frame[:, width // 2:])):
            msg = self.bridge.cv2_to_imgmsg(half.copy(), encoding="bgr8")
            msg.header.stamp = stamp
            msg.header.frame_id = "camera"
            pub.publish(msg)

    def destroy_node(self):
        self.cap.release()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = FramePublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
