#!/usr/bin/env python3
# in: camera/left/image_raw (Image) | out: perception/segmented_image (Image), perception/centroid (PointStamped, px), cmd/steering, cmd/throttle (Float64)
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


class SegmentModelNode(Node):
    def __init__(self):
        super().__init__("segment_model")
        model_path = self.declare_parameter("model_path", "model_proc.static_int8.onnx").value
        self.input_name = self.declare_parameter("input_name", "input").value
        self.throttle = self.declare_parameter("throttle", 0.43).value

        self.sess = rt.InferenceSession(
            model_path, providers=[("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"})])
        self.pid = PID(K_P, 0.00, 0, setpoint=0)
        self.prev_steer = 0

        self.bridge = CvBridge()
        self.create_subscription(Image, "camera/left/image_raw", self.run, qos_profile_sensor_data)
        self.image_pub = self.create_publisher(Image, "perception/segmented_image", qos_profile_sensor_data)
        self.centroid_pub = self.create_publisher(PointStamped, "perception/centroid", 10)
        self.steer_pub = self.create_publisher(Float64, "cmd/steering", 10)
        self.throt_pub = self.create_publisher(Float64, "cmd/throttle", 10)

    def run(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        img = cv2.resize(img, (640, 360))[4:-4, :, :]  # model wants 352x640
        img_rs = img.copy()

        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR HWC -> RGB CHW
        img = np.expand_dims(np.ascontiguousarray(img), 0).astype(np.float32) / 255.0
        da_out, ll_out = self.sess.run(None, {self.input_name: img})[:2]

        # da = drivable area, ll = lane lines
        DA = np.argmax(da_out, 1).astype(np.uint8)[0] * 255
        LL = np.argmax(ll_out, 1).astype(np.uint8)[0] * 255
        DA[:50, :] = 0  # model has a weird band of segmentation at the very top
        LL[:50, :] = 0
        img_rs[DA > 100] = [255, 0, 0]
        img_rs[LL > 100] = [0, 255, 0]

        # centroid of the largest drivable blob sets the PID target
        height, width, _ = img_rs.shape
        img_bin = cv2.inRange(img_rs, (240, 0, 0), (255, 0, 0))
        contours, _ = cv2.findContours(img_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cx = cy = 0.0
        if contours:
            M = cv2.moments(max(contours, key=cv2.contourArea))
            if M["m00"] != 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
            cv2.circle(img_rs, (int(cx), int(cy)), 5, (0, 255, 0), -1)
            self.pid.setpoint = OFFSET_MULTIPLIER * ((cx - width // 2) / width)

        steering = self.pid(self.prev_steer)

        out = self.bridge.cv2_to_imgmsg(img_rs, encoding="bgr8")
        out.header = msg.header
        self.image_pub.publish(out)

        point = PointStamped()
        point.header = msg.header
        point.point.x = float(cx)
        point.point.y = float(cy)
        self.centroid_pub.publish(point)

        self.steer_pub.publish(Float64(data=float(steering)))
        self.throt_pub.publish(Float64(data=float(self.throttle)))


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
