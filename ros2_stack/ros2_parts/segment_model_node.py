#!/usr/bin/env python3
"""ROS 2 port of parts/segment_model.py.

Runs the segmentation model on each frame and calculates steering from the
centroid of the detected track, using a PID to follow it.

The part's offset-to-throttle curve is not carried over: it computed one and
then overrode it with a constant throttle, so it never took effect.

subscribes: camera/left/image_raw (Image)
publishes:  perception/segmented_image (Image),
            perception/centroid (PointStamped, pixels),
            cmd/steering, cmd/throttle (Float64)
"""

import cv2
import numpy as np
import onnxruntime as rt
from simple_pid import PID
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Image
from std_msgs.msg import Float64

from ros2_parts.constants import K_P, OFFSET_MULTIPLIER
from ros2_parts.parameters import declare

# size the model expects, before the left/right trim
MODEL_SIZE = (640, 360)

# pixels trimmed off the top and bottom to reach the model's input height
VERTICAL_TRIM = 4

# the new model has a weird band of segmentation at the very top
TOP_CROP_ROWS = 50


def reduce_green(img):
    b, g, r = cv2.split(img)
    mask = g > 150
    g[mask] = (g[mask] * 0.7).astype(np.uint8)
    img = cv2.merge([b, g, r])
    return img


class SegmentModelNode(Node):
    def __init__(self):
        super().__init__("segment_model")

        model_path = declare(
            self, "model_path", "model_proc.static_int8.onnx", "ONNX model to load.")
        self.input_name = declare(
            self, "input_name", "input", "Name of the model's input tensor.")
        use_cuda = declare(self, "use_cuda", True, "Run on the GPU.")
        kp = declare(self, "kp", float(K_P), "Proportional gain on the centroid offset.")
        self.throttle = declare(self, "throttle", 0.43, "Constant throttle published.")

        if use_cuda:
            providers = [('CUDAExecutionProvider', {"cudnn_conv_algo_search": "DEFAULT"})]
        else:
            providers = ["CPUExecutionProvider"]
        self.sess = rt.InferenceSession(model_path, providers=providers)
        self.pid = PID(kp, 0.00, 0, setpoint=0)
        self.prev_steer = 0
        self.get_logger().info("model loaded")

        self.bridge = CvBridge()
        self.image_pub = self.create_publisher(
            Image, "perception/segmented_image", qos_profile_sensor_data)
        self.centroid_pub = self.create_publisher(PointStamped, "perception/centroid", 10)
        self.steer_pub = self.create_publisher(Float64, "cmd/steering", 10)
        self.throt_pub = self.create_publisher(Float64, "cmd/throttle", 10)
        self.create_subscription(
            Image, "camera/left/image_raw", self.on_image, qos_profile_sensor_data)

    def run(self, img):
        """
        Run segmentation model on the inputted image and calculate steering
        based on the centroid in the detected track. Uses the PID to follow
        the centroid of the track.

        Args:
            img (numpy.ndarray): Inputted image from the frame publisher
        Returns:
            img_rs (numpy.ndarray): Segmented image from the model after
                cropping and contour detection
            (contour_center_x, contour_center_y) (int, int): X and Y pixel
                values of the detected centroid
            steering (float): PID output from the location of the centroid
            throttle (float): constant throttle
        """
        img = cv2.resize(img, MODEL_SIZE)
        # cut 4 pixels off each top and bottom side & reshape to 352x640x3
        img = img[VERTICAL_TRIM:-VERTICAL_TRIM, :, :]
        img_rs = img.copy()

        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = np.expand_dims(img, 0)  # add a batch dimension
        img = img.astype(np.float32)
        img = img / 255.0

        # Run segmentation model on inputted image
        img_out = self.sess.run(None, {self.input_name: img})

        x0 = img_out[0]
        x1 = img_out[1]

        # Detect driveable area and lane lines

        # da = driveable area
        # ll = lane lines
        da_predict = np.argmax(x0, 1)
        ll_predict = np.argmax(x1, 1)

        height, width, _ = img_rs.shape
        DA = da_predict.astype(np.uint8)[0]*255
        LL = ll_predict.astype(np.uint8)[0]*255
        # crop the very top pixels
        DA[:TOP_CROP_ROWS, :] = 0
        LL[:TOP_CROP_ROWS, :] = 0
        img_rs[DA > 100] = [255, 0, 0]
        img_rs[LL > 100] = [0, 255, 0]

        image_center_x = width // 2
        # Create binary mask based on pixels in the range 240 to 255
        img_bin = cv2.inRange(img_rs, (240, 0, 0), (255, 0, 0))

        # Detect contours in the binary image
        contours, _ = cv2.findContours(img_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contour_center_x = contour_center_y = 0
        # Detect the centroid of the road - which should be the largest contour
        if len(contours) >= 1:
            # Find the largest contour by area
            largest_contour = max(contours, key=cv2.contourArea)

            # Find moments of the road and calculate centroid
            # M["m00"]: area of the object
            # M["m10"] and M["m01"]: intrinsic moment values based on the shape
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:  # To avoid division by zero
                contour_center_x = M["m10"] / M["m00"]
                contour_center_y = M["m01"] / M["m00"]
            else:
                contour_center_x = contour_center_y = 0

            # Draw the contour center using moments
            cv2.circle(
                img_rs, (int(contour_center_x), int(contour_center_y)), 5, (0, 255, 0), -1
            )

            # Calculate offset based on the X component of the centroid
            offset_x = contour_center_x - image_center_x
            scaled_offset_x = OFFSET_MULTIPLIER * (offset_x / width) # 2

            # Update the PID setpoint to the current scaled offset
            self.pid.setpoint = scaled_offset_x

            self.get_logger().debug(
                f"Contour center X: {contour_center_x}, "
                f"Image center X: {image_center_x}, "
                f"Offset X (scaled): {scaled_offset_x}")
        else:
            self.get_logger().debug("No contours found")

        # Update the steering value based on the previous steering value
        steering = self.pid(self.prev_steer)
        throttle = self.throttle
        return img_rs, (contour_center_x, contour_center_y), steering, throttle

    def on_image(self, msg):
        img_rs, centroid, steering, throttle = self.run(
            self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8"))

        out = self.bridge.cv2_to_imgmsg(img_rs, encoding="bgr8")
        out.header = msg.header
        self.image_pub.publish(out)

        point = PointStamped()
        point.header = msg.header
        point.point.x = float(centroid[0])
        point.point.y = float(centroid[1])
        self.centroid_pub.publish(point)

        self.steer_pub.publish(Float64(data=float(steering)))
        self.throt_pub.publish(Float64(data=float(throttle)))


def main(args=None):
    rclpy.init(args=args)
    node = SegmentModelNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
