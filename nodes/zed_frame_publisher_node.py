#!/usr/bin/env python3
# in: ZED camera (SDK) | out: camera/left/image_raw, camera/right/image_raw, camera/depth/image_raw (Image), zed_calibration_params.bin
import pickle

import pyzed.sl as sl
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

# used instead of the SDK's own calibration, which translate_node reads from disk
hard_coded = {
    "fx": 701.12,
    "fy": 701.12,
    "cx": 610.83,
    "cy": 380.3405,
    "k1": -0.175609,
    "k2": 0.0273627,
    "p1": 0.000324635,
    "p2": 0.00135292,
    "k3": 0.0,
}


class ZedFramePublisherNode(Node):
    def __init__(self):
        super().__init__("zed_frame_publisher")
        rate_hz = self.declare_parameter("rate_hz", 30.0).value

        self.zed = sl.Camera()
        init = sl.InitParameters()
        init.camera_resolution = sl.RESOLUTION.HD720
        init.depth_mode = sl.DEPTH_MODE.ULTRA
        init.coordinate_units = sl.UNIT.METER
        init.depth_minimum_distance = 0.5
        err = self.zed.open(init)
        if err != sl.ERROR_CODE.SUCCESS:
            self.zed.close()
            raise RuntimeError(f"Could not open the ZED camera: {err!r}")
        self.runtime = sl.RuntimeParameters()

        with open("zed_calibration_params.bin", "wb") as f:
            pickle.dump(hard_coded, f)

        self.bridge = CvBridge()
        self.left_pub = self.create_publisher(Image, "camera/left/image_raw", qos_profile_sensor_data)
        self.right_pub = self.create_publisher(Image, "camera/right/image_raw", qos_profile_sensor_data)
        self.depth_pub = self.create_publisher(Image, "camera/depth/image_raw", qos_profile_sensor_data)
        self.create_timer(1.0 / rate_hz, self.run)

    def run(self):
        if self.zed.grab(self.runtime) != sl.ERROR_CODE.SUCCESS:
            return
        left = sl.Mat()
        self.zed.retrieve_image(left, sl.VIEW.LEFT)
        right = sl.Mat()
        self.zed.retrieve_image(right, sl.VIEW.RIGHT)
        depth = sl.Mat()
        self.zed.retrieve_measure(depth, sl.MEASURE.DEPTH)

        stamp = self.get_clock().now().to_msg()
        for pub, mat, encoding in ((self.left_pub, left, "bgra8"),
                                   (self.right_pub, right, "bgra8"),
                                   (self.depth_pub, depth, "32FC1")):
            msg = self.bridge.cv2_to_imgmsg(mat.get_data(), encoding=encoding)
            msg.header.stamp = stamp
            msg.header.frame_id = "zed_left_camera_optical_frame"
            pub.publish(msg)

    def destroy_node(self):
        self.zed.close()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ZedFramePublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
