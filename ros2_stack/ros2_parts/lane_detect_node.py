#!/usr/bin/env python3
"""ROS 2 port of parts/lane_detect.py.

Adaptive threshold, Hough lines, then a Bezier fit to the two largest lane
contours. The midline between them is published as a Path.

The part also took a depth frame and camera intrinsics, but its body never
read them, so this node subscribes only to the color image.

subscribes: camera/left/image_raw (Image)
publishes:  perception/midline (Path, pixels),
            perception/lane_image (Image, the midline drawn)
"""

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from sensor_msgs.msg import Image

from ros2_parts.curve_fit_node import get_bezier, plot_bezier
from ros2_parts.parameters import declare

# points sampled along each fitted curve
CURVE_SAMPLES = 40

# every Nth contour pixel is fed to the fit, to keep it cheap
CONTOUR_STRIDE = 8


class LaneDetectNode(Node):
    def __init__(self):
        super().__init__("lane_detect")

        self.blockSizeGaus = declare(
            self, "block_size_gaussian", 117, "Adaptive threshold block size, odd.")
        self.constantGaus = declare(
            self, "constant_gaussian", -17, "Adaptive threshold constant.")
        self.closing_iterations = declare(
            self, "closing_iterations", 1, "Open/close iterations on the threshold.")
        self.kernel_size = declare(self, "kernel_size", 3, "Morphology kernel size.")
        self.crop_top = declare(self, "crop_top", 350, "First row kept.")
        self.crop_bottom = declare(self, "crop_bottom", 550, "Last row kept.")
        self.frame_id = declare(
            self, "frame_id", "camera", "Frame the midline is expressed in.")

        self.bridge = CvBridge()
        self.midline_pub = self.create_publisher(Path, "perception/midline", 10)
        self.image_pub = self.create_publisher(
            Image, "perception/lane_image", qos_profile_sensor_data)
        self.create_subscription(
            Image, "camera/left/image_raw", self.on_image, qos_profile_sensor_data)

    def get_bezier_curve(self, contour, x_shift):
        contour_points = np.transpose(np.nonzero(contour))[0::CONTOUR_STRIDE]
        control_points = np.array(get_bezier(contour_points))
        t = np.linspace(0, 1, CURVE_SAMPLES)
        curve = plot_bezier(t, control_points)
        curve = np.flip(curve, axis=1)
        curve[:, 0] += x_shift
        return curve

    def find_two_largest_contours(self, contours):
        """
        Returns the two largest contours in the list of contours.
        """
        largest_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
        return largest_contours

    def crop_to_contour(self, img, contour):
        """
        Returns an image cropped to the contour.
        """
        x, y, w, h = cv2.boundingRect(contour)
        rect = np.intp(cv2.boxPoints(cv2.minAreaRect(contour)))
        crop = img[y:y+h, x:x+w]
        mask = np.zeros_like(crop)
        rect = rect - np.array([x, y])
        cv2.drawContours(mask, [rect], 0, 255, -1)
        return cv2.bitwise_and(crop, mask)

    def kernelx(self, x):
        """
        Returns a square kernel of size x by x.
        """
        return np.ones((x, x), np.uint8)

    def gaussian_threshold(self, img, blockSize, constant):
        """
        Returns an image thresholded using adaptive gaussian thresholding.
        """
        return cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, blockSize, constant)

    def crop_image_top(self, img):
        """
        Returns an image cropped from the top.
        """
        rows, cols = img.shape[:2]
        return img[self.crop_top:self.crop_bottom, 0:cols]

    def run(self, in_image_rgb):
        img_normal = in_image_rgb

        img = cv2.cvtColor(img_normal, cv2.COLOR_BGR2GRAY)

        # Crop image to reduce value range and remove sky/background
        cropped_image = self.crop_image_top(img)
        # Gaussian Thresholding
        gaussian = self.gaussian_threshold(
            cropped_image, self.blockSizeGaus, self.constantGaus)

        opening = cv2.morphologyEx(
            gaussian, cv2.MORPH_OPEN, self.kernelx(self.kernel_size),
            iterations=self.closing_iterations)
        openclose = cv2.morphologyEx(
            opening, cv2.MORPH_CLOSE, self.kernelx(self.kernel_size),
            iterations=self.closing_iterations)

        linesP2 = cv2.HoughLinesP(
            openclose, 1, np.pi / 180, 50, None, minLineLength=60, maxLineGap=40)
        lines = np.zeros_like(openclose)

        if linesP2 is not None:
            for i in range(0, len(linesP2)):
                l = linesP2[i][0]
                cv2.line(lines, (l[0], l[1]), (l[2], l[3]), (255, 255, 255), 3, cv2.LINE_AA)

        special_kernel = np.array([[0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0],
                                   [0, 1, 1, 1, 0], [0, 1, 1, 1, 0]], np.uint8)

        ret, lines = cv2.threshold(lines, 127, 255, cv2.THRESH_BINARY)
        lines = cv2.dilate(lines, special_kernel, iterations=1)
        lines_dilated = cv2.bitwise_or(lines, openclose)

        open_open = cv2.morphologyEx(
            lines_dilated, cv2.MORPH_OPEN, self.kernelx(3), iterations=2)

        sobel1 = cv2.Sobel(open_open, cv2.CV_8UC1, 1, 0, ksize=3)

        contours_open_open, _ = cv2.findContours(
            open_open, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if len(contours_open_open) >= 1:
            cv2.drawContours(cropped_image, contours_open_open, -1, (0, 255, 0), 5)
            for contour in contours_open_open:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(cropped_image, (x, y), (x+w, y+h), (0, 0, 255), 1)

        largest = self.find_two_largest_contours(contours_open_open)

        if len(largest) == 0:
            return None, img_normal

        contour1 = self.crop_to_contour(sobel1, largest[0])
        if len(largest) == 1:
            self.get_logger().debug("Only one contour")
            curve1 = self.get_bezier_curve(contour1, cv2.boundingRect(largest[0])[0])
            return curve1, img_normal

        contour2 = self.crop_to_contour(sobel1, largest[1])

        curve1 = self.get_bezier_curve(contour1, cv2.boundingRect(largest[0])[0])
        curve2 = self.get_bezier_curve(contour2, cv2.boundingRect(largest[1])[0])

        midpoint_line = (curve1 + curve2) / 2

        height_adjust_midpoint_line = midpoint_line.copy()
        height_adjust_midpoint_line[:, 1] += self.crop_top

        cv2.polylines(img_normal, [np.int32(height_adjust_midpoint_line)],
                      isClosed=False, color=(255, 255, 255), thickness=2)

        return midpoint_line, img_normal

    def on_image(self, msg):
        midpoint_line, img_normal = self.run(
            self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8"))

        out = self.bridge.cv2_to_imgmsg(img_normal, encoding="bgr8")
        out.header = msg.header
        self.image_pub.publish(out)

        if midpoint_line is None:
            self.get_logger().debug("No contours found")
            return

        path = Path()
        path.header = msg.header
        path.header.frame_id = self.frame_id
        for point in midpoint_line:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = float(point[0])
            pose.pose.position.y = float(point[1]) + self.crop_top
            path.poses.append(pose)
        self.midline_pub.publish(path)


def main(args=None):
    rclpy.init(args=args)
    node = LaneDetectNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
