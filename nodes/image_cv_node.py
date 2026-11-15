#!/usr/bin/env python3
# in: camera/image_raw (Image) | out: perception/detection_image (Image), perception/object_centroid (PointStamped, px), perception/contour_area (Float64)
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Image
from std_msgs.msg import Float64


class ImageCvNode(Node):
    def __init__(self):
        super().__init__("image_cv")
        self.bridge = CvBridge()
        self.create_subscription(Image, "camera/image_raw", self.run, qos_profile_sensor_data)
        self.image_pub = self.create_publisher(Image, "perception/detection_image", qos_profile_sensor_data)
        self.centroid_pub = self.create_publisher(PointStamped, "perception/object_centroid", 10)
        self.area_pub = self.create_publisher(Float64, "perception/contour_area", 10)

    def run(self, msg):
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return

        # largest contour and its centroid
        maxContour = max(contours, key=cv2.contourArea)
        maxArea = cv2.contourArea(maxContour)
        M = cv2.moments(maxContour)
        if M["m00"] == 0:
            return
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        cv2.drawContours(image, [maxContour], 0, (255, 0, 0), 2)
        cv2.circle(image, (cX, cY), 5, (0, 0, 255), -1)

        out = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
        out.header = msg.header
        self.image_pub.publish(out)

        centroid = PointStamped()
        centroid.header = msg.header
        centroid.point.x = float(cX)
        centroid.point.y = float(cY)
        self.centroid_pub.publish(centroid)
        self.area_pub.publish(Float64(data=float(maxArea)))


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
