#!/usr/bin/env python3
# in: gps/fix (NavSatFix) | out: gps/pose (PoseStamped, east/north meters + travel heading), map->base_link tf if publish_tf
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped, TransformStamped
from sensor_msgs.msg import NavSatFix, NavSatStatus
from tf2_ros import TransformBroadcaster

from ros2_parts.orientation import yaw_to_quaternion

EARTH_RADIUS_M = 6378137.0  # WGS84 equatorial radius


class GpsToXyNode(Node):
    def __init__(self):
        super().__init__("gps_to_xy")
        ref_lat_deg = self.declare_parameter("ref_lat_deg", 40.4237).value
        ref_lon_deg = self.declare_parameter("ref_lon_deg", -86.9212).value
        self.min_heading_step_m = self.declare_parameter("min_heading_step", 0.3).value
        publish_tf = self.declare_parameter("publish_tf", False).value

        self.ref_lat_rad = math.radians(ref_lat_deg)
        self.ref_lon_rad = math.radians(ref_lon_deg)
        self.cos_ref_lat = math.cos(self.ref_lat_rad)

        self.prev_x_east_m = None
        self.prev_y_north_m = None
        self.gps_yaw_rad = 0.0

        self.create_subscription(NavSatFix, "gps/fix", self.run, qos_profile_sensor_data)
        self.pub = self.create_publisher(PoseStamped, "gps/pose", 10)
        self.tf_broadcaster = TransformBroadcaster(self) if publish_tf else None

    def to_xy(self, lat_deg, lon_deg):
        dlat = math.radians(lat_deg) - self.ref_lat_rad
        dlon = math.radians(lon_deg) - self.ref_lon_rad
        return dlon * self.cos_ref_lat * EARTH_RADIUS_M, dlat * EARTH_RADIUS_M

    def update_gps_yaw(self, x_east_m, y_north_m):
        # heading from travel direction; hold the old yaw if we barely moved
        if self.prev_x_east_m is not None:
            dx = x_east_m - self.prev_x_east_m
            dy = y_north_m - self.prev_y_north_m
            if math.hypot(dx, dy) >= self.min_heading_step_m:
                self.gps_yaw_rad = math.atan2(dy, dx)
        self.prev_x_east_m = x_east_m
        self.prev_y_north_m = y_north_m

    def run(self, fix):
        if fix.status.status == NavSatStatus.STATUS_NO_FIX:
            return
        x_east_m, y_north_m = self.to_xy(fix.latitude, fix.longitude)
        self.update_gps_yaw(x_east_m, y_north_m)

        pose = PoseStamped()
        pose.header.stamp = fix.header.stamp
        pose.header.frame_id = "map"
        pose.pose.position.x = x_east_m
        pose.pose.position.y = y_north_m
        pose.pose.orientation = yaw_to_quaternion(self.gps_yaw_rad)
        self.pub.publish(pose)

        if self.tf_broadcaster is not None:
            tf = TransformStamped()
            tf.header = pose.header
            tf.child_frame_id = "base_link"
            tf.transform.translation.x = x_east_m
            tf.transform.translation.y = y_north_m
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
