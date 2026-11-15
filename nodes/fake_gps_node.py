#!/usr/bin/env python3
# in: none | out: gps/fix (NavSatFix), gps/fix_type (String) -- noisy circle around origin
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import NavSatFix, NavSatStatus
from std_msgs.msg import String

R_EARTH = 6378137.0
RAD2DEG = 180.0 / np.pi


class FakeGpsNode(Node):
    def __init__(self):
        super().__init__("fake_gps")
        self.origin_lat = self.declare_parameter("origin_lat", 40.4237).value
        self.origin_lon = self.declare_parameter("origin_lon", -86.9212).value
        gps_rate = self.declare_parameter("gps_rate", 10.0).value
        speed_mps = self.declare_parameter("speed_mps", 2.0).value
        self.turn_radius_m = self.declare_parameter("turn_radius_m", 8.0).value
        self.gps_noise_std_m = self.declare_parameter("gps_noise_std_m", 1.0).value
        seed = self.declare_parameter("seed", 123).value

        self.rng = np.random.default_rng(seed)
        self.gps_dt = 1.0 / gps_rate
        self.theta = 0.0
        self.yaw_rate = speed_mps / self.turn_radius_m

        self.fix_pub = self.create_publisher(NavSatFix, "gps/fix", qos_profile_sensor_data)
        self.fix_type_pub = self.create_publisher(String, "gps/fix_type", 10)
        self.create_timer(self.gps_dt, self.run)

    def _meters_to_latlon(self, x_east_m, y_north_m):
        cos_lat0 = np.cos(np.radians(self.origin_lat))
        lat = self.origin_lat + (y_north_m / R_EARTH) * RAD2DEG
        lon = self.origin_lon + (x_east_m / (R_EARTH * cos_lat0)) * RAD2DEG
        return lat, lon

    def run(self):
        self.theta += self.yaw_rate * self.gps_dt
        x_east = self.turn_radius_m * np.cos(self.theta) + self.rng.normal(0.0, self.gps_noise_std_m)
        y_north = self.turn_radius_m * np.sin(self.theta) + self.rng.normal(0.0, self.gps_noise_std_m)
        lat, lon = self._meters_to_latlon(x_east, y_north)

        fix = NavSatFix()
        fix.header.stamp = self.get_clock().now().to_msg()
        fix.header.frame_id = "gps"
        fix.status.status = NavSatStatus.STATUS_FIX
        fix.latitude = float(lat)
        fix.longitude = float(lon)
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
