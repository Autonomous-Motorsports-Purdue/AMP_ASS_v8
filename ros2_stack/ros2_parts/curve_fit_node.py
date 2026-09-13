#!/usr/bin/env python3
"""ROS 2 port of parts/curve_fit.py.

Fits Bezier curves to the track edges and picks a waypoint between them. The
morphology, the RANSAC-fitted edges and the waypoint selection are unchanged.

The part also took the lane-line mask, but its body never read it, so this
node subscribes only to the drivable-area mask.

subscribes: perception/drivable_mask (Image, mono8)
publishes:  perception/waypoint (PointStamped, pixels),
            perception/curve_image (Image, the fit drawn)
"""

from math import factorial

import cv2
import numpy as np
from skimage.measure import ransac
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Image

from ros2_parts.parameters import declare

# points sampled along each fitted curve
CURVE_SAMPLES = 40

# every Nth edge pixel is fed to the fit, to keep it cheap
EDGE_STRIDE = 8

# waypoints above this row are above the horizon and cannot be driven to
HORIZON_ROW = 380


def comb(n, k):
    """
    Returns the combination of n choose k.
    """
    return factorial(n) / factorial(k) / factorial(n - k)


def normalize_path_length(points):
    """
    Returns a list of the normalized path length of the points. This
    parameterizes the curve by distance traveled rather than by index, so
    unevenly spaced points still fit smoothly.
    """
    path_length = [0]
    x, y = points[:, 0], points[:, 1]

    # calculate the path length
    for i in range(1, len(points)):
        path_length.append(
            np.sqrt((x[i] - x[i - 1])**2 + (y[i] - y[i - 1])**2) + path_length[i - 1])

    # normalize the path length
    # computes the percentage of path length at each point
    pct_len = []
    for i in range(len(path_length)):
        if (path_length[i] == 0):
            pct_len.append(0.01)
            continue
        pct_len.append(path_length[i] / path_length[-1])

    return pct_len


def get_bezier(points):
    """
    Returns the control points of a bezier curve.
    """
    num_points = len(points)

    x, y = points[:, 0], points[:, 1]

    # bezier matrix for a cubic curve
    bezier_matrix = np.array(
        [[-1, 3, -3, 1], [3, -6, 3, 0], [-3, 3, 0, 0], [1, 0, 0, 0]])
    bezier_inverse = np.linalg.inv(bezier_matrix)

    normalized_length = normalize_path_length(points)

    points_matrix = np.zeros((num_points, 4))

    for i in range(num_points):
        points_matrix[i] = [
            normalized_length[i]**3, normalized_length[i]**2, normalized_length[i], 1]

    points_transpose = points_matrix.transpose()
    square_points = np.matmul(points_transpose, points_matrix)

    if (np.linalg.det(square_points) == 0):
        square_inverse = np.linalg.pinv(square_points)
    else:
        square_inverse = np.linalg.inv(square_points)

    # solve for the solution matrix
    solution = np.matmul(np.matmul(bezier_inverse, square_inverse), points_transpose)

    # solve for the control points
    control_points_x = np.matmul(solution, x)
    control_points_y = np.matmul(solution, y)

    return list(zip(control_points_x, control_points_y))


def plot_bezier(t, cp):
    """
    Plots a bezier curve.
    t is the time values for the curve.
    cp is the control points of the curve.
    return is a tuple of the x and y values of the curve.
    """
    cp = np.array(cp)
    num_points, d = np.shape(cp)   # Number of points, Dimension of points
    num_points = num_points - 1
    curve = np.zeros((len(t), d))

    for i in range(num_points+1):
        # Bernstein polynomial
        val = comb(num_points, i) * t**i * (1.0-t)**(num_points-i)
        curve += np.outer(val, cp[i])

    return curve


def get_bezier_curve(line):
    points = np.transpose(np.nonzero(line))[0::EDGE_STRIDE]
    control_points = np.array(get_bezier(points))
    t = np.linspace(0, 1, CURVE_SAMPLES)
    curve = plot_bezier(t, control_points)
    curve = np.flip(curve, axis=1)
    return curve


class BezierRansacModel:
    def __init__(self, seed_pt=None):
        self.control_points = None
        self.seed = seed_pt

    def estimate(self, data):
        # data: (M,2) array of sample points
        if self.seed is not None:
            # data is an (m,2) array of x,y
            if not any((data == self.seed).all(axis=1)):
                return False   # reject this sample
        # otherwise fit as before
        self.control_points = get_bezier(data)
        return True

    def residuals(self, data):
        # for each data point, compute its closest distance to the fitted curve
        ts = np.linspace(0, 1, 500)                      # dense sampling
        curve_pts = plot_bezier(ts, self.control_points)  # (500,2)
        # compute minimal Euclidean distance from each data point to any curve sample
        d = np.linalg.norm(data[:, None, :] - curve_pts[None, :, :], axis=2)
        return np.min(d, axis=1)  # shape (M,)


class CurveFitNode(Node):
    def __init__(self):
        super().__init__("curve_fit")

        self.top_crop_rows = declare(
            self, "top_crop_rows", 50, "Rows blacked out at the top of the mask.")
        self.horizon_row = declare(
            self, "horizon_row", HORIZON_ROW, "Row the waypoint must sit below.")
        self.residual_threshold = declare(
            self, "residual_threshold", 10.0, "RANSAC inlier distance, in pixels.")

        self.bridge = CvBridge()
        self.waypoint_pub = self.create_publisher(PointStamped, "perception/waypoint", 10)
        self.curve_pub = self.create_publisher(
            Image, "perception/curve_image", qos_profile_sensor_data)
        self.create_subscription(
            Image, "perception/drivable_mask", self.on_mask, qos_profile_sensor_data)

    def ransac(self, image, left):
        ys, xs = np.nonzero(image)
        points = np.column_stack([xs, ys])  # or [ys, xs], be consistent in fit/evaluate
        if len(points) < 7:
            self.get_logger().debug("Not enough points to fit a curve")
            return np.zeros_like(image, dtype=np.uint8)
        # run RANSAC, seeded on the lowest point in the image
        seed = points[np.argmin(points[:, 1])]
        # if there are no points in the left quarter of the image and left is True, return empty image
        if left and np.sum(points[:, 0] < image.shape[1] * 5 // 6) == 0:
            self.get_logger().debug("No points in left quarter of the image")
            return np.zeros_like(image, dtype=np.uint8)
        # if there are no points in the right quarter of the image and left is False, return empty image
        if not left and np.sum(points[:, 0] > image.shape[1] // 6) == 0:
            self.get_logger().debug("No points in right quarter of the image")
            return np.zeros_like(image, dtype=np.uint8)

        best_model, inliers = ransac(
            data=points,
            model_class=lambda: BezierRansacModel(seed_pt=seed),
            min_samples=5,          # need at least degree+1 points to fit
            residual_threshold=self.residual_threshold,
            max_trials=len(points) // 3
        )

        # recover the inlier points and make the clean mask
        inlier_pts = points[inliers].squeeze()
        clean_mask = np.zeros_like(image, dtype=np.uint8)
        clean_mask[inlier_pts[:, 1], inlier_pts[:, 0]] = 1

        return clean_mask

    def get_waypoints(self, mid):
        # Get the waypoints from the mid bezier curve
        # select the middle point, if not below horizon line, then go down one
        # and select again until below horizon line
        select = 1 * len(mid) // 2
        top = mid[select]
        while top[1] > self.horizon_row:
            select += 1
            if select >= len(mid) - 1:
                return mid[len(mid) - 1]
            top = mid[select]
        return top

    def track(self, img):
        # set top rows to black
        img[0:self.top_crop_rows, :] = 0
        kernel5 = np.ones((5, 5), np.uint8)
        kernel3 = np.ones((3, 3), np.uint8)

        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel5, iterations=5)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel5, iterations=2)

        contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]

        if len(contours) == 0:
            self.get_logger().debug("No contours found")
            return None, None

        # Select the largest contour
        c = max(contours, key=cv2.contourArea)

        M = cv2.moments(c)
        if M["m00"] != 0:  # To avoid division by zero
            contour_center_x = M["m10"] / M["m00"]
            contour_center_y = M["m01"] / M["m00"]
        else:
            contour_center_x = contour_center_y = 0

        rect = np.intp(cv2.boxPoints(cv2.minAreaRect(c)))
        mask = np.zeros_like(img)
        cv2.drawContours(mask, [rect], -1, (255), -1)
        img = cv2.bitwise_and(img, mask)

        mask = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel3, iterations=3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel5, iterations=3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel3, iterations=2)

        img = mask

        sobelLeft = cv2.Sobel(img, cv2.CV_8UC1, 1, 0, ksize=1)
        sobelLeft[:, :2] = 0
        ransac_left = self.ransac(sobelLeft, True)

        inverse = cv2.bitwise_not(img)
        sobelRight = cv2.Sobel(inverse, cv2.CV_8UC1, 1, 0, ksize=1)
        sobelRight[:, -2:] = 0

        ransac_right = self.ransac(sobelRight, False)

        if len(np.nonzero(ransac_left)[0]) == 0:
            self.get_logger().debug("No left curve found")
            y_vals = np.linspace(
                img.shape[0] // 2, img.shape[0] - 1, CURVE_SAMPLES, dtype=np.uint32)
            cluster_curve_left = np.column_stack((np.zeros(CURVE_SAMPLES), y_vals))
        else:
            cluster_curve_left = get_bezier_curve(ransac_left)

        if len(np.nonzero(ransac_right)[0]) == 0:
            self.get_logger().debug("No right curve found")
            y_vals = np.linspace(
                img.shape[0] // 2, img.shape[0] - 1, CURVE_SAMPLES, dtype=np.uint32)
            cluster_curve_right = np.column_stack(
                (np.full(CURVE_SAMPLES, img.shape[1] - 1), y_vals))
        else:
            cluster_curve_right = get_bezier_curve(ransac_right)

        cluster_curve_image = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
        cluster_curve_image[:, :, 0] = img
        cv2.polylines(cluster_curve_image, [np.int32(cluster_curve_left)],
                      isClosed=False, color=(255, 255, 0), thickness=2)
        cv2.polylines(cluster_curve_image, [np.int32(cluster_curve_right)],
                      isClosed=False, color=(0, 255, 0), thickness=2)

        cluster_mid = (cluster_curve_left + cluster_curve_right) / 2
        cv2.polylines(cluster_curve_image, [np.int32(cluster_mid)],
                      isClosed=False, color=(0, 0, 255), thickness=2)
        waypoint = self.get_waypoints(cluster_mid)
        # draw centroid on image
        cv2.circle(cluster_curve_image,
                   (int(contour_center_x), int(contour_center_y)), 5, (155, 0, 255), -1)
        # Draw the waypoint on the curve image
        cv2.circle(cluster_curve_image,
                   (int(waypoint[0]), int(waypoint[1])), 5, (0, 255, 255), -1)
        # draw horizon line
        cv2.line(cluster_curve_image, (0, int(img.shape[0] * 0.5)),
                 (img.shape[1], int(img.shape[0] * 0.5)), (255, 255, 255), 2)
        return np.array(waypoint), cluster_curve_image

    def run(self, track):
        waypoint, curve = self.track(track)
        return waypoint, curve

    def on_mask(self, msg):
        mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        waypoint, curve = self.run(mask)
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
