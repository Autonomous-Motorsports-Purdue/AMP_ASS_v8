#!/usr/bin/env python3
# in: perception/drivable_mask (Image, mono8) | out: perception/waypoint (PointStamped, px), perception/curve_image (Image)
from math import comb

import cv2
import numpy as np
from skimage.measure import ransac
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Image

CURVE_SAMPLES = 40
HORIZON_ROW = 380  # waypoint must sit below this row
BEZIER_MATRIX = np.array([[-1, 3, -3, 1], [3, -6, 3, 0], [-3, 3, 0, 0], [1, 0, 0, 0]])


def normalize_path_length(points):
    # fraction of total path length at each point, so uneven spacing still fits smoothly
    steps = np.hypot(np.diff(points[:, 0]), np.diff(points[:, 1]))
    path_length = np.concatenate(([0.0], np.cumsum(steps)))
    pct_len = path_length / path_length[-1] if path_length[-1] else path_length
    pct_len[path_length == 0] = 0.01
    return pct_len


def get_bezier(points):
    # least-squares cubic bezier control points
    t = normalize_path_length(points)
    points_matrix = np.column_stack((t ** 3, t ** 2, t, np.ones_like(t)))
    square_inverse = np.linalg.pinv(points_matrix.T @ points_matrix)
    solution = np.linalg.inv(BEZIER_MATRIX) @ square_inverse @ points_matrix.T
    return list(zip(solution @ points[:, 0], solution @ points[:, 1]))


def plot_bezier(t, cp):
    cp = np.array(cp)
    n = len(cp) - 1
    curve = np.zeros((len(t), cp.shape[1]))
    for i in range(n + 1):
        curve += np.outer(comb(n, i) * t ** i * (1.0 - t) ** (n - i), cp[i])  # Bernstein polynomial
    return curve


def get_bezier_curve(line):
    points = np.transpose(np.nonzero(line))[0::8]  # every 8th edge pixel keeps the fit cheap
    curve = plot_bezier(np.linspace(0, 1, CURVE_SAMPLES), get_bezier(points))
    return np.flip(curve, axis=1)


class BezierRansacModel:
    def __init__(self, seed_pt=None):
        self.control_points = None
        self.seed = seed_pt

    def estimate(self, data):
        # only accept samples that include the seed point
        if self.seed is not None and not any((data == self.seed).all(axis=1)):
            return False
        self.control_points = get_bezier(data)
        return True

    def residuals(self, data):
        # distance from each point to the nearest of 500 samples on the curve
        curve_pts = plot_bezier(np.linspace(0, 1, 500), self.control_points)
        d = np.linalg.norm(data[:, None, :] - curve_pts[None, :, :], axis=2)
        return np.min(d, axis=1)


class CurveFitNode(Node):
    def __init__(self):
        super().__init__("curve_fit")
        self.residual_threshold = self.declare_parameter("residual_threshold", 10.0).value

        self.bridge = CvBridge()
        self.create_subscription(Image, "perception/drivable_mask", self.run, qos_profile_sensor_data)
        self.waypoint_pub = self.create_publisher(PointStamped, "perception/waypoint", 10)
        self.curve_pub = self.create_publisher(Image, "perception/curve_image", qos_profile_sensor_data)

    def ransac(self, image, left):
        ys, xs = np.nonzero(image)
        points = np.column_stack([xs, ys])
        empty = np.zeros_like(image, dtype=np.uint8)
        if len(points) < 7:
            return empty
        # give up if there are no edge points on the expected side
        if left and np.sum(points[:, 0] < image.shape[1] * 5 // 6) == 0:
            return empty
        if not left and np.sum(points[:, 0] > image.shape[1] // 6) == 0:
            return empty

        seed = points[np.argmin(points[:, 1])]
        best_model, inliers = ransac(
            data=points,
            model_class=lambda: BezierRansacModel(seed_pt=seed),
            min_samples=5,
            residual_threshold=self.residual_threshold,
            max_trials=len(points) // 3)

        inlier_pts = points[inliers].squeeze()
        empty[inlier_pts[:, 1], inlier_pts[:, 0]] = 1
        return empty

    def get_waypoint(self, mid):
        # start at the middle of the midline and walk down until below the horizon
        select = len(mid) // 2
        while mid[select][1] > HORIZON_ROW:
            select += 1
            if select >= len(mid) - 1:
                return mid[-1]
        return mid[select]

    def track(self, img):
        img[0:50, :] = 0
        kernel5 = np.ones((5, 5), np.uint8)
        kernel3 = np.ones((3, 3), np.uint8)

        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel5, iterations=5)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel5, iterations=2)
        contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        if len(contours) == 0:
            return None, None

        # keep only the largest blob, clipped to its bounding box
        c = max(contours, key=cv2.contourArea)
        rect = np.intp(cv2.boxPoints(cv2.minAreaRect(c)))
        mask = np.zeros_like(img)
        cv2.drawContours(mask, [rect], -1, 255, -1)
        img = cv2.bitwise_and(img, mask)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel3, iterations=3)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel5, iterations=3)
        img = cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel3, iterations=2)

        # left edge = dark->light, right edge = light->dark
        sobelLeft = cv2.Sobel(img, cv2.CV_8UC1, 1, 0, ksize=1)
        sobelLeft[:, :2] = 0
        ransac_left = self.ransac(sobelLeft, True)
        sobelRight = cv2.Sobel(cv2.bitwise_not(img), cv2.CV_8UC1, 1, 0, ksize=1)
        sobelRight[:, -2:] = 0
        ransac_right = self.ransac(sobelRight, False)

        # a missing edge falls back to the image border
        h, w = img.shape
        y_vals = np.linspace(h // 2, h - 1, CURVE_SAMPLES)
        if np.any(ransac_left):
            curve_left = get_bezier_curve(ransac_left)
        else:
            curve_left = np.column_stack((np.zeros(CURVE_SAMPLES), y_vals))
        if np.any(ransac_right):
            curve_right = get_bezier_curve(ransac_right)
        else:
            curve_right = np.column_stack((np.full(CURVE_SAMPLES, w - 1), y_vals))

        mid = (curve_left + curve_right) / 2
        waypoint = self.get_waypoint(mid)

        curve_image = np.zeros((h, w, 3), np.uint8)
        curve_image[:, :, 0] = img
        cv2.polylines(curve_image, [np.int32(curve_left)], isClosed=False, color=(255, 255, 0), thickness=2)
        cv2.polylines(curve_image, [np.int32(curve_right)], isClosed=False, color=(0, 255, 0), thickness=2)
        cv2.polylines(curve_image, [np.int32(mid)], isClosed=False, color=(0, 0, 255), thickness=2)
        cv2.circle(curve_image, (int(waypoint[0]), int(waypoint[1])), 5, (0, 255, 255), -1)
        return waypoint, curve_image

    def run(self, msg):
        mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        waypoint, curve = self.track(mask)
        if waypoint is None:
            return

        point = PointStamped()
        point.header = msg.header
        point.point.x = float(waypoint[0])
        point.point.y = float(waypoint[1])
        self.waypoint_pub.publish(point)

        out = self.bridge.cv2_to_imgmsg(curve, encoding="bgr8")
        out.header = msg.header
        self.curve_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = CurveFitNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
