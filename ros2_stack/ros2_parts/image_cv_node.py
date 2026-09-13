#!/usr/bin/env python3
"""ROS 2 port of parts/image_cv.py.

The cv2.imshow preview is opt-in here, since a node usually runs without a
display.

subscribes: camera/image_raw (Image)
publishes:  perception/detection_image (Image),
            perception/object_centroid (PointStamped),
            perception/contour_area (Float64, pixels)
"""

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Image
from std_msgs.msg import Float64

from ros2_parts.parameters import declare


class ImageCvNode(Node):
    def __init__(self):
        super().__init__("image_cv")

        self.show_preview = declare(
            self, "show_preview", False, "Open an OpenCV window with the detection drawn.")

        self.bridge = CvBridge()
        self.image_pub = self.create_publisher(
            Image, "perception/detection_image", qos_profile_sensor_data)
        self.centroid_pub = self.create_publisher(
            PointStamped, "perception/object_centroid", 10)
        self.area_pub = self.create_publisher(Float64, "perception/contour_area", 10)
        self.create_subscription(
            Image, "camera/image_raw", self.on_image, qos_profile_sensor_data)

    def run(self, image):
        if image is not None:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Convert to binary - all pixels under 127 to 0, over 127 to 255
            ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

            # Detect Contours
            contours, hierarchy = cv2.findContours(
                binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Find Largest Contour
            maxContour = None
            maxArea = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > maxArea:
                    maxArea = area
                    maxContour = contour
            if maxContour is None:
                return None, None, None, None

            # Draw largest contour
            cv2.drawContours(image, [maxContour], 0, (255, 0, 0), 2)

            # Draw Contour centroid
            M = cv2.moments(maxContour)

            # Calculate the centroid of the contour
            if M["m00"] == 0:
                return None, None, None, None
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            # Draw centroid
            cv2.circle(image, (cX, cY), 5, (0, 0, 255), -1)
            if self.show_preview:
                cv2.imshow("Contour", image)
                cv2.waitKey(1)

            return image, cX, cY, maxArea

    def on_image(self, msg):
        image, cX, cY, maxArea = self.run(
            self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8"))
        if image is None:
            return

        out = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
        out.header = msg.header
        self.image_pub.publish(out)

        centroid = PointStamped()
        centroid.header = msg.header
        centroid.point.x = float(cX)
        centroid.point.y = float(cY)
        self.centroid_pub.publish(centroid)

        self.area_pub.publish(Float64(data=float(maxArea)))

    def destroy_node(self):
        if self.show_preview:
            cv2.destroyAllWindows()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ImageCvNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
