#!/usr/bin/env python3
"""ROS 2 port of parts/fake_gps.py.

Simulates GPS latitude/longitude for bench testing, so it stands in for
gps_node. Generates a circular path in local East/North meters around the
origin, with Gaussian position noise.

publishes: gps/fix (NavSatFix), gps/fix_type (String)
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import NavSatFix, NavSatStatus
from std_msgs.msg import String

from ros2_parts.parameters import declare

R_EARTH = 6378137.0
RAD2DEG = 180.0 / np.pi


class FakeGpsNode(Node):
    def __init__(self):
        super().__init__("fake_gps")

        self.origin_lat = declare(self, "origin_lat", 40.4237, "Circle center latitude.")
        self.origin_lon = declare(self, "origin_lon", -86.9212, "Circle center longitude.")
        gps_rate = declare(self, "gps_rate", 10.0, "Simulated fix rate, in Hz.")
        speed_mps = declare(self, "speed_mps", 2.0, "Simulated ground speed.")
        self.turn_radius_m = declare(
            self, "turn_radius_m", 8.0, "Radius of the circle driven, in meters.")
        self.gps_noise_std_m = declare(
            self, "gps_noise_std_m", 1.0, "Std dev of the position noise, in meters.")
        seed = declare(self, "seed", 123, "Seed for the noise generator.")
        self.frame_id = declare(self, "frame_id", "gps", "Frame the antenna sits in.")

        self.rng = np.random.default_rng(seed)
        self.gps_dt = 1.0 / gps_rate
        self.theta = 0.0
        self.yaw_rate = speed_mps / self.turn_radius_m

        self.fix_pub = self.create_publisher(NavSatFix, "gps/fix", qos_profile_sensor_data)
        self.fix_type_pub = self.create_publisher(String, "gps/fix_type", 10)
        self.create_timer(self.gps_dt, self.publish_fix)

    def _meters_to_latlon(self, x_east_m, y_north_m):
        cos_lat0 = np.cos(np.radians(self.origin_lat))
        cos_lat0 = np.clip(cos_lat0, 1e-6, None)
        lat = self.origin_lat + (y_north_m / R_EARTH) * RAD2DEG
        lon = self.origin_lon + (x_east_m / (R_EARTH * cos_lat0)) * RAD2DEG
        return lat, lon

    def run(self):
        self.theta += self.yaw_rate * self.gps_dt

        x_east = self.turn_radius_m * np.cos(self.theta)
        y_north = self.turn_radius_m * np.sin(self.theta)

        noise_e = self.rng.normal(0.0, self.gps_noise_std_m)
        noise_n = self.rng.normal(0.0, self.gps_noise_std_m)

        return self._meters_to_latlon(x_east + noise_e, y_north + noise_n)

    def publish_fix(self):
        lat, lon = self.run()

        fix = NavSatFix()
        fix.header.stamp = self.get_clock().now().to_msg()
        fix.header.frame_id = self.frame_id
        fix.status.status = NavSatStatus.STATUS_FIX
        fix.status.service = NavSatStatus.SERVICE_GPS
        fix.latitude = float(lat)
        fix.longitude = float(lon)
        fix.altitude = 0.0
        variance = self.gps_noise_std_m ** 2
        fix.position_covariance[0] = variance
        fix.position_covariance[4] = variance
        fix.position_covariance[8] = variance
        fix.position_covariance_type = NavSatFix.COVARIANCE_TYPE_DIAGONAL_KNOWN
        self.fix_pub.publish(fix)

        self.fix_type_pub.publish(String(data="RTK FIXED"))


def main(args=None):
    rclpy.init(args=args)
    node = FakeGpsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
