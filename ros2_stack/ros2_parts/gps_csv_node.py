#!/usr/bin/env python3
# in: recorded GPS csv (lat/lon columns required) | out: gps/fix (NavSatFix), gps/fix_type (String)
import csv

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import NavSatFix, NavSatStatus
from std_msgs.msg import String


def pick(row, keys, default=None):
    for k in keys:
        if row.get(k):
            return row[k]
    return default


class GpsCsvNode(Node):
    def __init__(self):
        super().__init__("gps_csv")
        path = self.declare_parameter("path", "").value
        playback_rate_hz = self.declare_parameter("playback_rate_hz", 4.0).value
        self.loop = self.declare_parameter("loop", True).value

        self.rows = []
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                lat = pick(row, ("lat_raw", "lat", "latitude"))
                lon = pick(row, ("lon_raw", "lon", "longitude"))
                if lat is None or lon is None:
                    continue
                alt = pick(row, ("alt", "altitude"), 0.0)
                fix = pick(row, ("fix",), "NO FIX")
                self.rows.append((float(lat), float(lon), float(alt), fix))
        self.idx = 0

        self.fix_pub = self.create_publisher(NavSatFix, "gps/fix", qos_profile_sensor_data)
        self.fix_type_pub = self.create_publisher(String, "gps/fix_type", 10)
        self.timer = self.create_timer(1.0 / playback_rate_hz, self.run)

    def run(self):
        lat, lon, alt, fix_type = self.rows[self.idx]
        self.idx += 1
        if self.idx >= len(self.rows):
            if self.loop:
                self.idx = 0
            else:
                self.timer.cancel()

        fix = NavSatFix()
        fix.header.stamp = self.get_clock().now().to_msg()
        fix.header.frame_id = "gps"
        if fix_type.strip() in ("RTK FLOAT", "RTK FIXED"):
            fix.status.status = NavSatStatus.STATUS_GBAS_FIX
        else:
            fix.status.status = NavSatStatus.STATUS_FIX
        fix.latitude = lat
        fix.longitude = lon
        fix.altitude = alt
        self.fix_pub.publish(fix)
        self.fix_type_pub.publish(String(data=fix_type))


def main(args=None):
    rclpy.init(args=args)
    node = GpsCsvNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
