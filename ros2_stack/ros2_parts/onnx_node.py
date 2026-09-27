#!/usr/bin/env python3
# in: camera/image_raw (Image) | out: perception/lane_mask, perception/drivable_mask (Image, mono8)
import numpy as np
import onnxruntime as rt
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


class OnnxNode(Node):
    def __init__(self):
        super().__init__("onnx")
        model_path = self.declare_parameter("model_path", "./model.onnx").value
        self.input_name = self.declare_parameter("input_name", "input.1").value
        use_cuda = self.declare_parameter("use_cuda", True).value

        if use_cuda:
            providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "EXHAUSTIVE"})]
        else:
            providers = ["CPUExecutionProvider"]
        self.sess = rt.InferenceSession(model_path, providers=providers)

        self.bridge = CvBridge()
        self.create_subscription(Image, "camera/image_raw", self.run, qos_profile_sensor_data)
        self.lane_pub = self.create_publisher(Image, "perception/lane_mask", qos_profile_sensor_data)
        self.drivable_pub = self.create_publisher(Image, "perception/drivable_mask", qos_profile_sensor_data)

    def run(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR HWC -> RGB CHW
        img = np.expand_dims(np.ascontiguousarray(img), 0).astype(np.float32) / 255.0

        da_out, ll_out = self.sess.run(None, {self.input_name: img})[:2]

        # da = drivable area, ll = lane lines
        drivable = np.argmax(da_out, 1).astype(np.uint8)[0] * 255
        drivable[500:, :] = 255  # road right in front of the kart is always drivable
        lanes = np.argmax(ll_out, 1).astype(np.uint8)[0] * 255

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
