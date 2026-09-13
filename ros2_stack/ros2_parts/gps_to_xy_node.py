#!/usr/bin/env python3
"""ROS 2 port of GPS_to_xy in parts/gps_to_xy.py.

Convert geodetic coordinates (lat/lon in degrees) to a local Cartesian frame.

Frame definition:
- x: East (+x is to the right on a north-up plot)
- y: North (+y is upward on a north-up plot)

This uses a local tangent-plane approximation around a reference
latitude/longitude. It is accurate for typical small- to medium-sized driving
areas. Heading comes from the direction of travel between fixes, so the
orientation stays identity until the kart has moved min_heading_step meters.

subscribes: gps/fix (NavSatFix)
publishes:  gps/pose (PoseStamped, east/north meters from the datum),
            map -> base_link on /tf when publish_tf is set
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped, TransformStamped
from sensor_msgs.msg import NavSatFix, NavSatStatus
from tf2_ros import TransformBroadcaster

from ros2_parts.orientation import yaw_to_quaternion
from ros2_parts.parameters import declare

EARTH_RADIUS_M = 6378137.0  # WGS84 equatorial radius


class GpsToXyNode(Node):
    def __init__(self):
        super().__init__("gps_to_xy")

        ref_lat_deg = declare(
            self, "ref_lat_deg", 40.4237, "Datum latitude of the local frame, in degrees.")
        ref_lon_deg = declare(
            self, "ref_lon_deg", -86.9212, "Datum longitude of the local frame, in degrees.")
        self.min_heading_step_m = declare(
            self, "min_heading_step", 0.3,
            "Travel required between fixes before the heading updates, in meters.")
        self.frame_id = declare(self, "frame_id", "map", "Frame the pose is expressed in.")
        self.child_frame_id = declare(
            self, "child_frame_id", "base_link", "Frame the transform moves.")
        publish_tf = declare(
            self, "publish_tf", False, "Broadcast the transform as well as the pose.")

        self.ref_lat_rad = math.radians(ref_lat_deg)
        self.ref_lon_rad = math.radians(ref_lon_deg)
        self.cos_ref_lat = math.cos(self.ref_lat_rad)
        self.get_logger().info(f"Local frame anchored at {ref_lat_deg:.7f}, {ref_lon_deg:.7f}")

        # Stateful heading estimate from consecutive local-frame displacements.
        self.prev_x_east_m = None
        self.prev_y_north_m = None
        self.gps_yaw_rad = None

        self.pub = self.create_publisher(PoseStamped, "gps/pose", 10)
        self.tf_broadcaster = TransformBroadcaster(self) if publish_tf else None
        self.create_subscription(NavSatFix, "gps/fix", self.on_fix, qos_profile_sensor_data)

    def to_xy(self, lat_deg, lon_deg):
        """Return (x, y) in meters where x=east and y=north."""
        dlat = math.radians(lat_deg) - self.ref_lat_rad
        dlon = math.radians(lon_deg) - self.ref_lon_rad

        x_east_m = dlon * self.cos_ref_lat * EARTH_RADIUS_M
        y_north_m = dlat * EARTH_RADIUS_M
        return x_east_m, y_north_m

    def _update_gps_yaw(self, x_east_m, y_north_m):
        """
        Update and return heading in radians from +x (east), CCW positive.

        If displacement is too small, keep the previous yaw to suppress jitter.
        """
        if self.prev_x_east_m is None or self.prev_y_north_m is None:
            self.prev_x_east_m = x_east_m
            self.prev_y_north_m = y_north_m
            return self.gps_yaw_rad

        dx = x_east_m - self.prev_x_east_m
        dy = y_north_m - self.prev_y_north_m
        step_m = math.hypot(dx, dy)

        if step_m >= self.min_heading_step_m:
            self.gps_yaw_rad = math.atan2(dy, dx)

        self.prev_x_east_m = x_east_m
        self.prev_y_north_m = y_north_m
        return self.gps_yaw_rad

    def run(self, lat_deg, lon_deg):
        x_east_m, y_north_m = self.to_xy(lat_deg, lon_deg)
        gps_yaw_rad = self._update_gps_yaw(x_east_m, y_north_m)
        return x_east_m, y_north_m, gps_yaw_rad

    def on_fix(self, fix):
        if fix.status.status == NavSatStatus.STATUS_NO_FIX:
            self.get_logger().warn("receiver reports no fix", throttle_duration_sec=5.0)
            return
        if not math.isfinite(fix.latitude) or not math.isfinite(fix.longitude):
            self.get_logger().warn("fix carries a non-finite coord", throttle_duration_sec=5.0)
            return

        x_east_m, y_north_m, gps_yaw_rad = self.run(fix.latitude, fix.longitude)

        pose = PoseStamped()
        pose.header.stamp = fix.header.stamp
        pose.header.frame_id = self.frame_id
        pose.pose.position.x = x_east_m
        pose.pose.position.y = y_north_m
        pose.pose.orientation = yaw_to_quaternion(gps_yaw_rad or 0.0)
        self.pub.publish(pose)

        if self.tf_broadcaster is not None:
            tf = TransformStamped()
            tf.header = pose.header
            tf.child_frame_id = self.child_frame_id
            tf.transform.translation.x = pose.pose.position.x
            tf.transform.translation.y = pose.pose.position.y
            tf.transform.rotation = pose.pose.orientation
            self.tf_broadcaster.sendTransform(tf)


def main(args=None):
    rclpy.init(args=args)
    node = GpsToXyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
