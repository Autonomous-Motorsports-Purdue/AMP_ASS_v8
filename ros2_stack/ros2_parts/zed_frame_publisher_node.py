#!/usr/bin/env python3
"""ROS 2 port of parts/zed_frame_publisher.py.

The camera is opened in HD720 with ULTRA depth in meters, as before, and the
calibration is still pickled to disk for translate_node to pick up.

publishes: camera/left/image_raw, camera/right/image_raw,
           camera/depth/image_raw (Image),
           camera/left/camera_info (CameraInfo)
"""

import pickle

import pyzed.sl as sl
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, Image

from ros2_parts.parameters import declare

# used when the SDK's own calibration is not trusted
hard_coded = {
    "fx": 701.12,
    "fy": 701.12,
    "cx": 610.83,
    "cy": 380.3405,
    "k1": -0.175609,
    "k2": 0.0273627,
    "p1": 0.000324635,
    "p2": 0.00135292,
    "k3": 0.0
}


class ZedFramePublisherNode(Node):
    def __init__(self):
        super().__init__("zed_frame_publisher")

        calibration_path = declare(
            self, "calibration_path", "zed_calibration_params.bin",
            "File the camera intrinsics are pickled to.")
        self.use_hard_coded = declare(
            self, "use_hard_coded_calibration", True,
            "Publish the hard-coded intrinsics instead of the SDK's own.")
        self.frame_id = declare(
            self, "frame_id", "zed_left_camera_optical_frame",
            "Frame the images are taken in.")
        min_depth_m = declare(
            self, "depth_minimum_distance", 0.5, "Nearest depth measured, in meters.")
        rate_hz = declare(self, "rate_hz", 30.0, "Rate frames are grabbed at, in Hz.")

        self.zed = sl.Camera()
        init = sl.InitParameters()
        init.camera_resolution = sl.RESOLUTION.HD720
        init.depth_mode = sl.DEPTH_MODE.ULTRA
        init.coordinate_units = sl.UNIT.METER
        init.depth_minimum_distance = min_depth_m

        err = self.zed.open(init)
        if err != sl.ERROR_CODE.SUCCESS:
            self.zed.close()
            raise RuntimeError(f"Could not open the ZED camera: {err!r}")

        self.runtime = sl.RuntimeParameters()
        self.calibration_params = self.read_calibration()

        # Dump to binary file
        with open(calibration_path, "wb") as f:
            pickle.dump(self.calibration_params, f)

        self.get_logger().info("ZED camera connected")

        self.bridge = CvBridge()
        self.left_pub = self.create_publisher(
            Image, "camera/left/image_raw", qos_profile_sensor_data)
        self.right_pub = self.create_publisher(
            Image, "camera/right/image_raw", qos_profile_sensor_data)
        self.depth_pub = self.create_publisher(
            Image, "camera/depth/image_raw", qos_profile_sensor_data)
        self.info_pub = self.create_publisher(
            CameraInfo, "camera/left/camera_info", qos_profile_sensor_data)
        self.create_timer(1.0 / rate_hz, self.publish_frames)

    def read_calibration(self):
        left_cam = (self.zed.get_camera_information()
                    .camera_configuration.calibration_parameters.left_cam)

        # Access intrinsic parameters
        calibration_params = {
            "fx": left_cam.fx,          # Focal length in x
            "fy": left_cam.fy,          # Focal length in y
            "cx": left_cam.cx,          # Principal point x
            "cy": left_cam.cy,          # Principal point y
            "k1": left_cam.disto[0],    # Radial distortion coefficient 1
            "k2": left_cam.disto[1],    # Radial distortion coefficient 2
            "k3": left_cam.disto[2],    # Radial distortion coefficient 3
            "p1": left_cam.disto[3],    # Tangential distortion coefficient 1
            "p2": left_cam.disto[5],    # Tangential distortion coefficient 2
        }

        self.get_logger().info(f"ZED reports {calibration_params}")
        return hard_coded if self.use_hard_coded else calibration_params

    def camera_info(self, stamp, width, height):
        c = self.calibration_params
        info = CameraInfo()
        info.header.stamp = stamp
        info.header.frame_id = self.frame_id
        info.width = width
        info.height = height
        info.distortion_model = "plumb_bob"
        info.d = [c["k1"], c["k2"], c["p1"], c["p2"], c["k3"]]
        info.k = [c["fx"], 0.0, c["cx"],
                  0.0, c["fy"], c["cy"],
                  0.0, 0.0, 1.0]
        info.p = [c["fx"], 0.0, c["cx"], 0.0,
                  0.0, c["fy"], c["cy"], 0.0,
                  0.0, 0.0, 1.0, 0.0]
        return info

    def run(self):
        if self.zed.grab(self.runtime) == sl.ERROR_CODE.SUCCESS:
            left = sl.Mat()
            self.zed.retrieve_image(left, sl.VIEW.LEFT)
            right = sl.Mat()
            self.zed.retrieve_image(right, sl.VIEW.RIGHT)
            depth = sl.Mat()
            self.zed.retrieve_measure(depth, sl.MEASURE.DEPTH)

            return left, right, depth

        return None, None, None

    def publish_frames(self):
        left, right, depth = self.run()
        if left is None:
            self.get_logger().warn("no frame from the ZED", throttle_duration_sec=5.0)
            return

        stamp = self.get_clock().now().to_msg()

        for pub, mat, encoding in (
            (self.left_pub, left, "bgra8"),
            (self.right_pub, right, "bgra8"),
            (self.depth_pub, depth, "32FC1"),
        ):
            msg = self.bridge.cv2_to_imgmsg(mat.get_data(), encoding=encoding)
            msg.header.stamp = stamp
            msg.header.frame_id = self.frame_id
            pub.publish(msg)

        self.info_pub.publish(
            self.camera_info(stamp, left.get_width(), left.get_height()))

    def destroy_node(self):
        if self.zed is not None:
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
