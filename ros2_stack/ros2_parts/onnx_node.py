#!/usr/bin/env python3
"""ROS 2 port of parts/onnx.py.

subscribes: camera/image_raw (Image)
publishes:  perception/lane_mask, perception/drivable_mask (Image, mono8)
"""

import numpy as np
import onnxruntime as rt
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from ros2_parts.parameters import declare

# row below which drivable is forced on -- the road under the nose is always road
DRIVABLE_FLOOR_ROW = 500


class OnnxNode(Node):
    def __init__(self):
        super().__init__("onnx")

        model_path = declare(self, "model_path", "./model.onnx", "ONNX model to load.")
        self.input_name = declare(
            self, "input_name", "input.1", "Name of the model's input tensor.")
        use_cuda = declare(self, "use_cuda", True, "Run on the GPU.")

        if use_cuda:
            providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "EXHAUSTIVE"})]
        else:
            providers = ["CPUExecutionProvider"]
        self.sess = rt.InferenceSession(model_path, providers=providers)
        self.get_logger().info("model loaded")

        self.bridge = CvBridge()
        self.lane_pub = self.create_publisher(
            Image, "perception/lane_mask", qos_profile_sensor_data)
        self.drivable_pub = self.create_publisher(
            Image, "perception/drivable_mask", qos_profile_sensor_data)
        self.create_subscription(
            Image, "camera/image_raw", self.on_image, qos_profile_sensor_data)

    def run(self, img):
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = np.expand_dims(img, 0)  # add a batch dimension
        img = img.astype(np.float32)
        img = img / 255.0

        img_out = self.sess.run(None, {self.input_name: img})

        x0 = img_out[0]
        x1 = img_out[1]

        # da = driveable area
        # ll = lane lines
        da_predict = np.argmax(x0, 1)
        ll_predict = np.argmax(x1, 1)

        drivable = da_predict.astype(np.uint8)[0] * 255
        drivable[DRIVABLE_FLOOR_ROW:, :] = 255
        lanes = ll_predict.astype(np.uint8)[0] * 255

        return lanes, drivable

    def on_image(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        lanes, drivable = self.run(img)

        for pub, mask in ((self.lane_pub, lanes), (self.drivable_pub, drivable)):
            out = self.bridge.cv2_to_imgmsg(mask, encoding="mono8")
            out.header = msg.header
            pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = OnnxNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
