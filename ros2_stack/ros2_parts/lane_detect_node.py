#!/usr/bin/env python3
# in: camera/left/image_raw (Image) | out: perception/midline (Path, px), perception/lane_image (Image)
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

SPECIAL_KERNEL = np.array([[0, 1, 1, 1, 0]] * 5, np.uint8)


def kernelx(x):
    return np.ones((x, x), np.uint8)


def crop_to_contour(img, contour):
    x, y, w, h = cv2.boundingRect(contour)
    rect = np.intp(cv2.boxPoints(cv2.minAreaRect(contour))) - np.array([x, y])
    crop = img[y:y + h, x:x + w]
    mask = np.zeros_like(crop)
    cv2.drawContours(mask, [rect], 0, 255, -1)
    return cv2.bitwise_and(crop, mask)


def get_bezier_curve(contour, x_shift):
    contour_points = np.transpose(np.nonzero(contour))[0::8]  # every 8th pixel keeps the fit cheap
    control_points = np.array(get_bezier(contour_points))
    curve = np.flip(plot_bezier(np.linspace(0, 1, 40), control_points), axis=1)
    curve[:, 0] += x_shift
    return curve


class LaneDetectNode(Node):
    def __init__(self):
        super().__init__("lane_detect")
        self.blockSizeGaus = 117
        self.constantGaus = -17
        self.closing_iterations = 1
        self.kernel_size = 3
        self.crop_top = 350
        self.crop_bottom = 550

        self.bridge = CvBridge()
        self.create_subscription(Image, "camera/left/image_raw", self.run, qos_profile_sensor_data)
        self.midline_pub = self.create_publisher(Path, "perception/midline", 10)
        self.image_pub = self.create_publisher(Image, "perception/lane_image", qos_profile_sensor_data)

    def run(self, msg):
        img_normal = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        img = cv2.cvtColor(img_normal, cv2.COLOR_BGR2GRAY)

        # crop out sky and hood, then threshold
        cropped_image = img[self.crop_top:self.crop_bottom, :]
        gaussian = cv2.adaptiveThreshold(cropped_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, self.blockSizeGaus, self.constantGaus)
        opening = cv2.morphologyEx(gaussian, cv2.MORPH_OPEN, kernelx(self.kernel_size),
                                   iterations=self.closing_iterations)
        openclose = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernelx(self.kernel_size),
                                     iterations=self.closing_iterations)

        # reinforce straight segments with Hough lines
        linesP2 = cv2.HoughLinesP(openclose, 1, np.pi / 180, 50, None, minLineLength=60, maxLineGap=40)
        lines = np.zeros_like(openclose)
        if linesP2 is not None:
            for l in linesP2[:, 0]:
                cv2.line(lines, (l[0], l[1]), (l[2], l[3]), 255, 3, cv2.LINE_AA)
        ret, lines = cv2.threshold(lines, 127, 255, cv2.THRESH_BINARY)
        lines = cv2.dilate(lines, SPECIAL_KERNEL, iterations=1)
        open_open = cv2.morphologyEx(cv2.bitwise_or(lines, openclose), cv2.MORPH_OPEN, kernelx(3), iterations=2)

        sobel1 = cv2.Sobel(open_open, cv2.CV_8UC1, 1, 0, ksize=3)
        contours, _ = cv2.findContours(open_open, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        largest = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
        if not largest:
            return

        # midline is the average of the two lane curves (or the one curve if only one found)
        curves = [get_bezier_curve(crop_to_contour(sobel1, c), cv2.boundingRect(c)[0]) for c in largest]
        midpoint_line = sum(curves) / len(curves)
        midpoint_line[:, 1] += self.crop_top

        cv2.polylines(img_normal, [np.int32(midpoint_line)], isClosed=False, color=(255, 255, 255), thickness=2)
        out = self.bridge.cv2_to_imgmsg(img_normal, encoding="bgr8")
        out.header = msg.header
        self.image_pub.publish(out)

        path = Path()
        path.header = msg.header
        for point in midpoint_line:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = float(point[0])
            pose.pose.position.y = float(point[1])
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
