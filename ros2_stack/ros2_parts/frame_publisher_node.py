#!/usr/bin/env python3
"""ROS 2 port of parts/frame_publisher.py.

The ZED is opened as one wide frame and cut down the middle, which is how it
presents itself over plain UVC without the SDK.

publishes: camera/left/image_raw, camera/right/image_raw (Image)
"""

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from ros2_parts.parameters import declare


class FramePublisherNode(Node):
    def __init__(self):
        super().__init__("frame_publisher")

        device = declare(self, "device", 0, "OpenCV capture device index.")
        width = declare(self, "width", 2560, "Capture width of the combined frame.")
        height = declare(self, "height", 720, "Capture height of the combined frame.")
        self.frame_id = declare(self, "frame_id", "camera", "Frame the images are taken in.")
        rate_hz = declare(self, "rate_hz", 30.0, "Rate frames are grabbed at, in Hz.")

        self.cap = cv2.VideoCapture(device)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            self.get_logger().error(f"Could not open capture device {device}")
        self.frame = None

        self.bridge = CvBridge()
        self.left_pub = self.create_publisher(
            Image, "camera/left/image_raw", qos_profile_sensor_data)
        self.right_pub = self.create_publisher(
            Image, "camera/right/image_raw", qos_profile_sensor_data)
        self.create_timer(1.0 / rate_hz, self.publish_frames)

    def grab_middle_section(self, image, width_ratio, height_ratio):
        """
        Grabs the middle section of an image based on given ratios.
        """
        height, width = image.shape[:2]

        start_x = int((1 - width_ratio) / 2 * width)
        end_x = int(start_x + width_ratio * width)
        start_y = int((1 - height_ratio) / 2 * height)
        end_y = int(start_y + height_ratio * height)

        return image[start_y:end_y, start_x:end_x]

    def remove_green(self, img, green_factor, min_green):
        arr = img.astype(np.float32)
        R = arr[..., 0]
        G = arr[..., 1]
        B = arr[..., 2]

        avg_RB = (R + B) / 2.0
        mask = (G > min_green) & (G > green_factor * avg_RB)
        # Broadcast white color to all masked pixels
        arr[mask] = [255.0, 255.0, 255.0]

        return arr.astype(np.uint8)

    def run(self):
        """
        Read in and return the current left and right frames from the ZED.
        """
        ret, self.frame = self.cap.read()
        if self.frame is not None:
            height, width = self.frame.shape[:2]

            # Split the self.frame in half
            # Assuming we want to split vertically
            left_half = self.frame[:, :width // 2]
            right_half = self.frame[:, width // 2:]
            return np.array(left_half), np.array(right_half)

        return None, None

    def publish_frames(self):
        left_half, right_half = self.run()
        if left_half is None:
            self.get_logger().warn("no frame from the camera", throttle_duration_sec=5.0)
            return

        stamp = self.get_clock().now().to_msg()
        for pub, half in ((self.left_pub, left_half), (self.right_pub, right_half)):
            msg = self.bridge.cv2_to_imgmsg(half, encoding="bgr8")
            msg.header.stamp = stamp
            msg.header.frame_id = self.frame_id
            pub.publish(msg)

    def destroy_node(self):
        if self.cap is not None:
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
