#!/usr/bin/env python3
# in: GNSS receiver serial + NTRIP corrections | out: gps/fix (NavSatFix), gps/vel (TwistStamped, ENU), gps/fix_type (String)
import math
from queue import Empty, Queue
from threading import Event

from serial import Serial

from pygnssutils.globals import FORMAT_PARSED
from pygnssutils.gnssntripclient import GNSSNTRIPClient
from pygnssutils.gnssstreamer import GNSSStreamer
from pygnssutils.helpers import parse_url

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import NavSatFix, NavSatStatus
from std_msgs.msg import String

KNOTS_TO_MPS = 0.514444


def safe_float(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


class GpsNode(Node):
    def __init__(self):
        super().__init__("gps")
        port = self.declare_parameter("port", "/dev/ttyACM0").value
        ntrip_url = self.declare_parameter("ntrip_url", "108.59.49.226:9000/MSM4_NEAR").value
        ntrip_user = self.declare_parameter("ntrip_user", "automp1").value
        ntrip_password = self.declare_parameter("ntrip_password", "automp1").value
        rate_hz = self.declare_parameter("rate_hz", 30.0).value

        self.lat = None
        self.lon = None
        self.alt = None
        self.hdop = None
        self.speed_mps = None
        self.course_deg = None

        # pygnssutils runs the receiver and NTRIP client on its own threads and fills gnss_queue
        self.gnss_queue = Queue()
        self.stop_event = Event()
        self.ser = Serial(port, 9600, timeout=3)
        self.gnss = GNSSStreamer(
            self, self.ser, outformat=FORMAT_PARSED, validate=1, msgmode=0, parsebitfield=1,
            quitonerror=1, protfilter=7, msgfilter="GNGGA,GPGGA,GNRMC,GPRMC,GNVTG,GPVTG",
            limit=0, outqueue=self.gnss_queue, stopevent=self.stop_event)
        self.gnss.run()

        if not ntrip_url.startswith("http"):
            ntrip_url = f"http://{ntrip_url}"
        protocol, hostname, ntrip_port, mountpoint = parse_url(ntrip_url)
        self.ntrip = GNSSNTRIPClient(self)
        self.ntrip.run(
            server=hostname, port=ntrip_port, https=1 if protocol == "https" else 0,
            mountpoint=mountpoint, ntripuser=ntrip_user, ntrippassword=ntrip_password,
            version="2.0", ggamode=0, ggainterval=1, datatype="RTCM",
            output=self.ser, stopevent=self.stop_event)

        self.fix_pub = self.create_publisher(NavSatFix, "gps/fix", qos_profile_sensor_data)
        self.vel_pub = self.create_publisher(TwistStamped, "gps/vel", qos_profile_sensor_data)
        self.fix_type_pub = self.create_publisher(String, "gps/fix_type", 10)
        self.create_timer(1.0 / rate_hz, self.run)

    def drain_queue(self):
        while True:
            try:
                data = self.gnss_queue.get_nowait()
            except Empty:
                return
            identity = str(getattr(data, "identity", ""))

            if identity.endswith("GGA"):
                self.lat = safe_float(getattr(data, "lat", None), self.lat)
                self.lon = safe_float(getattr(data, "lon", None), self.lon)
                self.alt = safe_float(getattr(data, "alt", None), self.alt)
                self.hdop = safe_float(getattr(data, "HDOP", None), self.hdop)
            elif identity.endswith("RMC"):
                self.lat = safe_float(getattr(data, "lat", None), self.lat)
                self.lon = safe_float(getattr(data, "lon", None), self.lon)
                spd_knots = safe_float(getattr(data, "spd", None))
                if spd_knots is not None:
                    self.speed_mps = spd_knots * KNOTS_TO_MPS
                self.course_deg = safe_float(getattr(data, "cog", None), self.course_deg)
            elif identity.endswith("VTG"):
                sogk = safe_float(getattr(data, "sogk", None))
                if sogk is not None:
                    self.speed_mps = sogk / 3.6
                self.course_deg = safe_float(getattr(data, "cogt", None), self.course_deg)

    def run(self):
        self.drain_queue()
        fix_type = str(self.gnss.get_coordinates().get("fix", "NO FIX"))
        self.fix_type_pub.publish(String(data=fix_type))
        if self.lat is None or self.lon is None:
            return

        stamp = self.get_clock().now().to_msg()
        fix = NavSatFix()
        fix.header.stamp = stamp
        fix.header.frame_id = "gps"
        if fix_type.strip() in ("RTK FLOAT", "RTK FIXED"):
            fix.status.status = NavSatStatus.STATUS_GBAS_FIX
        else:
            fix.status.status = NavSatStatus.STATUS_FIX
        fix.latitude = self.lat
        fix.longitude = self.lon
        fix.altitude = self.alt or 0.0
        if self.hdop is not None:
            # rough: 1 m of error per unit of HDOP
            fix.position_covariance[0] = self.hdop ** 2
            fix.position_covariance[4] = self.hdop ** 2
            fix.position_covariance[8] = 2.0 * self.hdop ** 2
            fix.position_covariance_type = NavSatFix.COVARIANCE_TYPE_APPROXIMATED
        self.fix_pub.publish(fix)

        # course is noise when nearly stopped
        if self.course_deg is not None and self.speed_mps is not None and self.speed_mps > 0.5:
            theta = math.radians(self.course_deg)
            vel = TwistStamped()
            vel.header.stamp = stamp
            vel.header.frame_id = "gps"
            vel.twist.linear.x = self.speed_mps * math.sin(theta)  # east
            vel.twist.linear.y = self.speed_mps * math.cos(theta)  # north
            self.vel_pub.publish(vel)

    def destroy_node(self):
        self.stop_event.set()
        self.ntrip.stop()
        self.gnss.stop()
        self.ser.close()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = GpsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
