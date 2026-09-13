#!/usr/bin/env python3
"""ROS 2 port of parts/gps_csv.py.

Replays GPS fixes from a CSV file in place of the live receiver. The CSV is
expected to have a header row. Columns are matched by name with sensible
fallbacks; only lat/lon are required. Missing optional columns are emitted as
default values ("NO FIX", 0, 0.0, 0).

publishes: the same topics as gps_node minus velocity, so it is a drop-in
           stand-in for the receiver
"""

import csv

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import NavSatFix, NavSatStatus
from std_msgs.msg import Float64, String, UInt16

from ros2_parts.parameters import declare

# fix wordings that mean an RTK correction was being applied
RTK_FIX_TYPES = ("RTK FLOAT", "RTK FIXED")


class GpsCsvNode(Node):

    _LAT_KEYS = ("lat_raw", "lat", "latitude")
    _LON_KEYS = ("lon_raw", "lon", "longitude")
    _ALT_KEYS = ("alt", "altitude")
    _FIX_KEYS = ("fix",)
    _AGE_KEYS = ("corr_age", "diffage", "diff_age")
    _HDOP_KEYS = ("hdop",)
    _SAT_KEYS = ("sat_count", "numSV", "sip")

    def __init__(self):
        super().__init__("gps_csv")

        path = declare(self, "path", "", "CSV of recorded fixes to replay.")
        playback_rate_hz = declare(
            self, "playback_rate_hz", 4.0, "Rate fixes are replayed at, in Hz.")
        self.loop = declare(self, "loop", True, "Restart the file when it runs out.")
        self.frame_id = declare(self, "frame_id", "gps", "Frame the antenna sits in.")

        self.rows = self._load(path)
        if not self.rows:
            raise ValueError(f"GPS_CSV: no rows parsed from {path}")
        self.get_logger().info(f"Replaying {len(self.rows)} fixes from {path}")

        self.idx = 0
        self.latest = self.rows[0]

        self.fix_pub = self.create_publisher(NavSatFix, "gps/fix", qos_profile_sensor_data)
        self.fix_type_pub = self.create_publisher(String, "gps/fix_type", 10)
        self.hdop_pub = self.create_publisher(Float64, "gps/hdop", 10)
        self.satellites_pub = self.create_publisher(UInt16, "gps/satellites", 10)
        self.correction_age_pub = self.create_publisher(Float64, "gps/correction_age", 10)

        self.timer = self.create_timer(1.0 / playback_rate_hz, self.publish_next)

    @classmethod
    def _pick(cls, row, keys, default, cast):
        for k in keys:
            if k in row and row[k] not in (None, ""):
                try:
                    return cast(row[k])
                except (ValueError, TypeError):
                    return default
        return default

    @classmethod
    def _parse_row(cls, row):
        lat = cls._pick(row, cls._LAT_KEYS, None, float)
        lon = cls._pick(row, cls._LON_KEYS, None, float)
        if lat is None or lon is None:
            return None
        alt = cls._pick(row, cls._ALT_KEYS, 0.0, float)
        fix = cls._pick(row, cls._FIX_KEYS, "NO FIX", str)
        corr_age = cls._pick(row, cls._AGE_KEYS, 0, int)
        hdop = cls._pick(row, cls._HDOP_KEYS, 0.0, float)
        sat_count = cls._pick(row, cls._SAT_KEYS, 0, int)
        return (lat, lon, alt, fix, corr_age, hdop, sat_count)

    @classmethod
    def _load(cls, path):
        rows = []
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for raw in reader:
                parsed = cls._parse_row(raw)
                if parsed is not None:
                    rows.append(parsed)
        return rows

    def _advance(self):
        self.latest = self.rows[self.idx]
        self.idx += 1
        if self.idx >= len(self.rows):
            if self.loop:
                self.idx = 0
            else:
                self.idx = len(self.rows) - 1
                self.timer.cancel()
                self.get_logger().info("reached the end of the recording")

    def run(self):
        self._advance()
        return self.latest

    def publish_next(self):
        lat, lon, alt, fix_type, corr_age, hdop, sat_count = self.run()

        fix = NavSatFix()
        fix.header.stamp = self.get_clock().now().to_msg()
        fix.header.frame_id = self.frame_id
        if str(fix_type).strip() in RTK_FIX_TYPES:
            fix.status.status = NavSatStatus.STATUS_GBAS_FIX
        else:
            fix.status.status = NavSatStatus.STATUS_FIX
        fix.status.service = NavSatStatus.SERVICE_GPS
        fix.latitude = lat
        fix.longitude = lon
        fix.altitude = alt
        fix.position_covariance_type = NavSatFix.COVARIANCE_TYPE_UNKNOWN
        self.fix_pub.publish(fix)

        self.fix_type_pub.publish(String(data=str(fix_type)))
        self.hdop_pub.publish(Float64(data=hdop))
        self.satellites_pub.publish(UInt16(data=sat_count))
        self.correction_age_pub.publish(Float64(data=float(corr_age)))


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
