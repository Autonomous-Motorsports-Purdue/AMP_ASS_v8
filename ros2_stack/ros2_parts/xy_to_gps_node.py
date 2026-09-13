#!/usr/bin/env python3
"""ROS 2 port of XY_to_GPS in parts/gps_to_xy.py.

Inverse of gps_to_xy_node: local east/north meters -> lat/lon degrees. Useful
for plotting the filtered pose on a map that expects lat/lon. The datum has to
match the one gps_to_xy_node is anchored at, or the output lands somewhere
else entirely.

subscribes: odometry/filtered (Odometry)
publishes:  gps/fused_fix (NavSatFix)
"""

import math

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import NavSatFix, NavSatStatus

from ros2_parts.parameters import declare

EARTH_RADIUS_M = 6378137.0  # WGS84 equatorial radius


class XyToGpsNode(Node):
    def __init__(self):
        super().__init__("xy_to_gps")

        ref_lat_deg = declare(
            self, "ref_lat_deg", 40.4237, "Datum latitude of the local frame, in degrees.")
        ref_lon_deg = declare(
            self, "ref_lon_deg", -86.9212, "Datum longitude of the local frame, in degrees.")
        self.altitude = declare(
            self, "altitude", 0.0, "Constant altitude stamped on the published fix.")
        self.frame_id = declare(self, "frame_id", "gps", "Frame the antenna sits in.")

        self.ref_lat_rad = math.radians(ref_lat_deg)
        self.ref_lon_rad = math.radians(ref_lon_deg)
        self.cos_ref_lat = math.cos(self.ref_lat_rad)
        self.get_logger().info(f"Local frame anchored at {ref_lat_deg:.7f}, {ref_lon_deg:.7f}")

        self.pub = self.create_publisher(NavSatFix, "gps/fused_fix", 10)
        self.create_subscription(Odometry, "odometry/filtered", self.on_odometry, 10)

    def to_latlon(self, x_east_m, y_north_m):
        """Return (lat_deg, lon_deg) from local east/north meters."""
        lat_rad = self.ref_lat_rad + (y_north_m / EARTH_RADIUS_M)
        lon_rad = self.ref_lon_rad + (x_east_m / (EARTH_RADIUS_M * self.cos_ref_lat))
        return math.degrees(lat_rad), math.degrees(lon_rad)

    def run(self, x, y):
        return self.to_latlon(x, y)

    def on_odometry(self, odom):
        p = odom.pose.pose.position
        if not math.isfinite(p.x) or not math.isfinite(p.y):
            self.get_logger().warn("odometry position is not finite", throttle_duration_sec=5.0)
            return

        lat_deg, lon_deg = self.run(p.x, p.y)

        fix = NavSatFix()
        fix.header.stamp = odom.header.stamp
        fix.header.frame_id = self.frame_id
        fix.status.status = NavSatStatus.STATUS_FIX
        fix.status.service = NavSatStatus.SERVICE_GPS
        fix.latitude = lat_deg
        fix.longitude = lon_deg
        fix.altitude = self.altitude
        # the filtered pose carries no geodetic uncertainty, so none is claimed
        fix.position_covariance_type = NavSatFix.COVARIANCE_TYPE_UNKNOWN
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
