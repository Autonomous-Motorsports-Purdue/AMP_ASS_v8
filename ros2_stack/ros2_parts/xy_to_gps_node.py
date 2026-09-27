#!/usr/bin/env python3
# in: odometry/filtered (Odometry) | out: gps/fused_fix (NavSatFix)
import math

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import NavSatFix, NavSatStatus

EARTH_RADIUS_M = 6378137.0  # WGS84 equatorial radius


class XyToGpsNode(Node):
    def __init__(self):
        super().__init__("xy_to_gps")
        # must match the datum gps_to_xy uses
        ref_lat_deg = self.declare_parameter("ref_lat_deg", 40.4237).value
        ref_lon_deg = self.declare_parameter("ref_lon_deg", -86.9212).value

        self.ref_lat_rad = math.radians(ref_lat_deg)
        self.ref_lon_rad = math.radians(ref_lon_deg)
        self.cos_ref_lat = math.cos(self.ref_lat_rad)

        self.create_subscription(Odometry, "odometry/filtered", self.run, 10)
        self.pub = self.create_publisher(NavSatFix, "gps/fused_fix", 10)

    def to_latlon(self, x_east_m, y_north_m):
        lat_rad = self.ref_lat_rad + (y_north_m / EARTH_RADIUS_M)
        lon_rad = self.ref_lon_rad + (x_east_m / (EARTH_RADIUS_M * self.cos_ref_lat))
        return math.degrees(lat_rad), math.degrees(lon_rad)

    def run(self, odom):
        p = odom.pose.pose.position
        fix = NavSatFix()
        fix.header.stamp = odom.header.stamp
        fix.header.frame_id = "gps"
        fix.status.status = NavSatStatus.STATUS_FIX
        fix.latitude, fix.longitude = self.to_latlon(p.x, p.y)
        self.pub.publish(fix)


def main(args=None):
    rclpy.init(args=args)
    node = XyToGpsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
