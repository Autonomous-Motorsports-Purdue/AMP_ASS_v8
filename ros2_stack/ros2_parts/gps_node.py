#!/usr/bin/env python3
"""ROS 2 port of parts/gps.py.

Starts the pygnssutils GNSS reader and NTRIP client once, then publishes the
latest parsed GNSS state. The receiver runs on a background thread owned by
pygnssutils; this node drains the queue it fills.

publishes: gps/fix (NavSatFix), gps/vel (TwistStamped, ENU ground velocity),
           gps/fix_type (String, the receiver's own wording such as
           "RTK FIXED", which NavSatStatus cannot express),
           gps/hdop, gps/satellites, gps/correction_age -- receiver quality
           figures with no standard message of their own
"""

import math
from queue import Empty, Queue
from threading import Event, Lock

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
from std_msgs.msg import Float64, String, UInt16

from ros2_parts.parameters import declare

# fix wordings that mean an RTK correction is being applied
RTK_FIX_TYPES = ("RTK FLOAT", "RTK FIXED")

# rough meters of position error per unit of HDOP, used for the covariance
HDOP_TO_METERS = 1.0


class GpsNode(Node):
    def __init__(self):
        super().__init__("gps")

        # Receiver configuration.
        self.serial_port = declare(self, "port", "/dev/ttyACM0", "Serial port the receiver is on.")
        self.baudrate = declare(self, "baudrate", 9600, "Serial baud rate.")
        self.timeout = declare(self, "timeout", 3, "Serial read timeout, in seconds.")
        self.frame_id = declare(self, "frame_id", "gps", "Frame the antenna sits in.")

        # NTRIP configuration.
        self.ntrip_url = declare(
            self, "ntrip_url", "108.59.49.226:9000/MSM4_NEAR", "NTRIP caster URL.")
        self.ntrip_user = declare(self, "ntrip_user", "automp1", "NTRIP username.")
        self.ntrip_password = declare(self, "ntrip_password", "automp1", "NTRIP password.")
        self.gga_interval = declare(
            self, "gga_interval", 1, "Seconds between GGA messages sent back to the caster.")
        self.debug = declare(self, "debug", False, "Log every parsed GNSS message.")
        rate_hz = declare(self, "rate_hz", 30.0, "Rate fixes are published at, in Hz.")

        self.msg_filter = "GNGGA,GPGGA,GNRMC,GPRMC,GNVTG,GPVTG"

        if not self.ntrip_url.startswith("http"):
            self.ntrip_url = f"http://{self.ntrip_url}"

        protocol, hostname, port, mountpoint = parse_url(self.ntrip_url)
        self.ntrip_settings = {
            "server": hostname,
            "port": port,
            "https": 1 if protocol == "https" else 0,
            "mountpoint": mountpoint,
            "ntripuser": self.ntrip_user,
            "ntrippassword": self.ntrip_password,
            "version": "2.0",
            "ggamode": 0,
            "ggainterval": self.gga_interval,
            "datatype": "RTCM",
        }

        self.gnss_queue = Queue()
        self.stop_event = Event()
        self.lock = Lock()
        self.started = False
        self.ser = None
        self.gnss = None
        self.ntrip = None
        self.latest_output = {
            "lat": None,
            "lon": None,
            "alt": None,
            "fix": "NO FIX",
            "fix_quality": None,
            "rtk_fixed": False,
            "diff_age": None,
            "diff_station": None,
            "hdop": None,
            "sip": None,
            "num_sv": None,
            "speed_mps": None,
            "speed_kph": None,
            "speed_knots": None,
            "course_deg": None,
            "course_valid": False,
            "vel_n_mps": None,
            "vel_e_mps": None,
        }

        self.fix_pub = self.create_publisher(NavSatFix, "gps/fix", qos_profile_sensor_data)
        self.vel_pub = self.create_publisher(TwistStamped, "gps/vel", qos_profile_sensor_data)
        self.fix_type_pub = self.create_publisher(String, "gps/fix_type", 10)
        self.hdop_pub = self.create_publisher(Float64, "gps/hdop", 10)
        self.satellites_pub = self.create_publisher(UInt16, "gps/satellites", 10)
        self.correction_age_pub = self.create_publisher(Float64, "gps/correction_age", 10)

        self._start_streams()
        self.create_timer(1.0 / rate_hz, self.publish_state)

    @staticmethod
    def safe_float(x, default=None):
        if x is None or x == "":
            return default
        try:
            return float(x)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_int(x, default=None):
        if x is None or x == "":
            return default
        try:
            return int(float(x))
        except (TypeError, ValueError):
            return default

    def _start_streams(self):
        with self.lock:
            if self.started:
                return

            self.ser = Serial(self.serial_port, self.baudrate, timeout=self.timeout)
            self.gnss = GNSSStreamer(
                self,
                self.ser,
                outformat=FORMAT_PARSED,
                validate=1,
                msgmode=0,
                parsebitfield=1,
                quitonerror=1,
                protfilter=7,
                msgfilter=self.msg_filter,
                limit=0,
                outqueue=self.gnss_queue,
                stopevent=self.stop_event,
            )
            self.ntrip = GNSSNTRIPClient(self)

            self.gnss.run()
            self.ntrip.run(
                **self.ntrip_settings,
                output=self.ser,
                stopevent=self.stop_event,
            )
            self.started = True
            self.get_logger().info(f"Streaming GNSS from {self.serial_port}")

    def get_coordinates(self):
        if self.gnss is None:
            return {
                "lat": 0.0,
                "lon": 0.0,
                "alt": 0.0,
                "sep": 0.0,
                "sip": 0,
                "fix": "NO FIX",
                "hdop": 0.0,
                "diffage": 0,
                "diffstation": 0,
            }

        status = self.gnss.get_coordinates()

        return {
            "lat": status.get("lat", 0.0),
            "lon": status.get("lon", 0.0),
            "alt": status.get("alt", 0.0),
            "sep": status.get("sep", 0.0),
            "sip": status.get("sip", 0),
            "fix": status.get("fix", "NO FIX"),
            "hdop": status.get("hdop", status.get("HDOP", status.get("hDOP", 0.0))),
            "diffage": status.get("diffage", status.get("diffAge", 0)),
            "diffstation": status.get("diffstation", status.get("diffStation", 0)),
        }

    def _update_course_velocity(self, out):
        """A course reported while nearly stationary is noise, so it is only
        trusted above half a meter per second."""
        course = out.get("course_deg")
        speed_mps = out.get("speed_mps")
        course_valid = (
            course is not None
            and speed_mps is not None
            and speed_mps > 0.5
        )
        out["course_valid"] = course_valid
        if not course_valid:
            out["vel_n_mps"] = None
            out["vel_e_mps"] = None
            return

        theta = math.radians(course)
        out["vel_n_mps"] = speed_mps * math.cos(theta)
        out["vel_e_mps"] = speed_mps * math.sin(theta)

    def _drain_gnss_queue(self):
        if self.gnss is None:
            return

        status = self.get_coordinates()

        while True:
            try:
                data = self.gnss_queue.get_nowait()
            except Empty:
                break

            if not hasattr(data, "identity"):
                continue

            identity = str(getattr(data, "identity", ""))
            if self.debug:
                self.get_logger().info(f"[GNSS MSG] {identity} {data}")

            with self.lock:
                out = dict(self.latest_output)
                out["fix"] = status.get("fix", out.get("fix"))

                if identity.endswith("GGA"):
                    lat = self.safe_float(getattr(data, "lat", None), out.get("lat"))
                    lon = self.safe_float(getattr(data, "lon", None), out.get("lon"))
                    alt = self.safe_float(getattr(data, "alt", None), out.get("alt"))
                    hdop = self.safe_float(
                        getattr(data, "HDOP", getattr(data, "hDOP", None)),
                        status.get("hdop", out.get("hdop")),
                    )
                    num_sv = self._safe_int(
                        getattr(data, "numSV", None), status.get("sip", out.get("num_sv")))
                    diff_age = self.safe_float(
                        getattr(data, "diffAge", None),
                        status.get("diffage", out.get("diff_age")))
                    diff_station = self._safe_int(
                        getattr(data, "diffStation", None),
                        status.get("diffstation", out.get("diff_station")))
                    quality = self._safe_int(
                        getattr(data, "quality", None), out.get("fix_quality"))

                    out["lat"] = lat
                    out["lon"] = lon
                    out["alt"] = alt
                    out["hdop"] = hdop
                    out["num_sv"] = num_sv
                    out["sip"] = num_sv
                    out["diff_age"] = diff_age
                    out["diff_station"] = diff_station
                    out["fix_quality"] = quality
                    out["rtk_fixed"] = quality == 4

                elif identity.endswith("RMC"):
                    lat = self.safe_float(getattr(data, "lat", None), out.get("lat"))
                    lon = self.safe_float(getattr(data, "lon", None), out.get("lon"))
                    spd_knots = self.safe_float(
                        getattr(data, "spd", None), out.get("speed_knots"))
                    cog = self.safe_float(getattr(data, "cog", None), out.get("course_deg"))

                    out["lat"] = lat
                    out["lon"] = lon
                    out["speed_knots"] = spd_knots
                    if spd_knots is not None:
                        speed_mps = spd_knots * 0.514444
                        out["speed_mps"] = speed_mps
                        out["speed_kph"] = speed_mps * 3.6
                    out["course_deg"] = cog

                elif identity.endswith("VTG"):
                    sogn = self.safe_float(getattr(data, "sogn", None), out.get("speed_knots"))
                    sogk = self.safe_float(getattr(data, "sogk", None), out.get("speed_kph"))
                    cogt = self.safe_float(getattr(data, "cogt", None), out.get("course_deg"))

                    out["speed_knots"] = sogn
                    out["speed_kph"] = sogk if sogk is not None else (
                        sogn * 1.852 if sogn is not None else out.get("speed_kph"))

                    if sogk is not None:
                        out["speed_mps"] = sogk / 3.6
                    elif sogn is not None:
                        out["speed_mps"] = sogn * 0.514444
                    out["course_deg"] = cogt

                self._update_course_velocity(out)
                self.latest_output = out

    def run(self):
        self._drain_gnss_queue()
        with self.lock:
            return dict(self.latest_output)

    def publish_state(self):
        out = self.run()

        stamp = self.get_clock().now().to_msg()
        fix_type = str(out["fix"])
        self.fix_type_pub.publish(String(data=fix_type))

        if out["hdop"] is not None:
            self.hdop_pub.publish(Float64(data=float(out["hdop"])))
        if out["num_sv"] is not None:
            self.satellites_pub.publish(UInt16(data=int(out["num_sv"])))
        if out["diff_age"] is not None:
            self.correction_age_pub.publish(Float64(data=float(out["diff_age"])))

        if out["lat"] is None or out["lon"] is None:
            self.get_logger().warn(
                f"no position yet (fix: {fix_type})", throttle_duration_sec=5.0)
            return

        fix = NavSatFix()
        fix.header.stamp = stamp
        fix.header.frame_id = self.frame_id
        if fix_type.strip() in RTK_FIX_TYPES:
            fix.status.status = NavSatStatus.STATUS_GBAS_FIX
        else:
            fix.status.status = NavSatStatus.STATUS_FIX
        fix.status.service = NavSatStatus.SERVICE_GPS
        fix.latitude = float(out["lat"])
        fix.longitude = float(out["lon"])
        fix.altitude = float(out["alt"] or 0.0)

        if out["hdop"] is not None:
            variance = (float(out["hdop"]) * HDOP_TO_METERS) ** 2
            fix.position_covariance[0] = variance
            fix.position_covariance[4] = variance
            fix.position_covariance[8] = 2.0 * variance
            fix.position_covariance_type = NavSatFix.COVARIANCE_TYPE_APPROXIMATED
        else:
            fix.position_covariance_type = NavSatFix.COVARIANCE_TYPE_UNKNOWN

        self.fix_pub.publish(fix)

        if out["course_valid"]:
            vel = TwistStamped()
            vel.header.stamp = stamp
            vel.header.frame_id = self.frame_id
            vel.twist.linear.x = float(out["vel_e_mps"])
            vel.twist.linear.y = float(out["vel_n_mps"])
            self.vel_pub.publish(vel)

    def destroy_node(self):
        self.stop_event.set()

        if self.ntrip is not None:
            self.ntrip.stop()

        if self.gnss is not None:
            self.gnss.stop()

        if self.ser is not None and self.ser.is_open:
            self.ser.close()

        with self.lock:
            self.started = False
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
