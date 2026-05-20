#!/usr/bin/env python3
"""
GNSS receiver part for Donkeycar.

This part starts the pygnssutils GNSS reader and NTRIP client once, then
returns the latest parsed GNSS state without blocking the vehicle loop.
"""

from queue import Empty, Queue
from threading import Event, Lock
from time import monotonic, sleep
import math

from serial import Serial

from pygnssutils.globals import FORMAT_PARSED
from pygnssutils.gnssntripclient import GNSSNTRIPClient
from pygnssutils.gnssstreamer import GNSSStreamer
from pygnssutils.helpers import parse_url


class GPS:
    def __init__(self, port: str, debug: bool = False, return_dict: bool = False):
        # Receiver configuration.
        self.serial_port = port
        self.baudrate = 9600
        self.timeout = 3

        # NTRIP configuration.
        self.ntrip_url = "108.59.49.226:9000/MSM4_NEAR"
        self.ntrip_user = "automp1"
        self.ntrip_password = "automp1"
        self.gga_interval = 1 # number of seconds between sending messages back up to NTRIP.
        self.msg_filter = "GNGGA,GPGGA,GNRMC,GPRMC,GNVTG,GPVTG"
        self.debug = debug
        self.return_dict = return_dict

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
            "last_msg_time": None,
            "last_update_monotonic": None,
        }

        # print(f"sleeping for 2")
        # sleep(2)
        # self.gnss_queue = Queue()

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

    def _to_legacy_tuple(self, out):
        return (
            out.get("lat"),
            out.get("lon"),
            out.get("alt"),
            out.get("fix"),
            out.get("diff_age"),
            out.get("hdop"),
            out.get("num_sv"),
            out.get("course_deg"),
            out.get("speed_mps")
        )

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
                print("[GNSS MSG]", identity, data)
                print("[FIELDS]", vars(data))

            with self.lock:
                out = dict(self.latest_output)
                out["fix"] = status.get("fix", out.get("fix"))
                out["last_update_monotonic"] = monotonic()
                msg_time = getattr(data, "time", None)
                out["last_msg_time"] = msg_time if msg_time not in ("", None) else out.get("last_msg_time")

                if identity.endswith("GGA"):
                    lat = self.safe_float(getattr(data, "lat", None), out.get("lat"))
                    lon = self.safe_float(getattr(data, "lon", None), out.get("lon"))
                    alt = self.safe_float(getattr(data, "alt", None), out.get("alt"))
                    hdop = self.safe_float(
                        getattr(data, "HDOP", getattr(data, "hDOP", None)),
                        status.get("hdop", out.get("hdop")),
                    )
                    num_sv = self._safe_int(getattr(data, "numSV", None), status.get("sip", out.get("num_sv")))
                    diff_age = self.safe_float(getattr(data, "diffAge", None), status.get("diffage", out.get("diff_age")))
                    diff_station = self._safe_int(getattr(data, "diffStation", None), status.get("diffstation", out.get("diff_station")))
                    quality = self._safe_int(getattr(data, "quality", None), out.get("fix_quality"))

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
                    spd_knots = self.safe_float(getattr(data, "spd", None), out.get("speed_knots"))
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
                    out["speed_kph"] = sogk if sogk is not None else (sogn * 1.852 if sogn is not None else out.get("speed_kph"))

                    if sogk is not None:
                        out["speed_mps"] = sogk / 3.6
                    elif sogn is not None:
                        out["speed_mps"] = sogn * 0.514444
                    out["course_deg"] = cogt

                self._update_course_velocity(out)
                self.latest_output = out

    def run(self):
        self._start_streams()
        self._drain_gnss_queue()
        with self.lock:
            return dict(self.latest_output) if self.return_dict else self._to_legacy_tuple(self.latest_output)

    def update(self):
        self._start_streams()
        while not self.stop_event.is_set():
            self._drain_gnss_queue()
            sleep(0.05)

    def run_threaded(self):
        self._drain_gnss_queue()

        with self.lock:
            print(f"[GPS]-{self._to_legacy_tuple(self.latest_output)}")
            return dict(self.latest_output) if self.return_dict else self._to_legacy_tuple(self.latest_output)

    def shutdown(self):
        self.stop_event.set()

        if self.ntrip is not None:
            self.ntrip.stop()

        if self.gnss is not None:
            self.gnss.stop()

        if self.ser is not None and self.ser.is_open:
            self.ser.close()

        with self.lock:
            self.started = False


def _format_output(sample):
    if isinstance(sample, tuple):
        lat, lon, alt, fix, diff_age, hdop, sip, heading = sample
        return (
            f"lat={lat}, lon={lon}, alt={alt}, fix={fix}, "
            f"diff_age={diff_age}, hdop={hdop}, sip={sip}, heading={heading}"
        )

    if sample is None:
        return "No sample"

    return (
        f"lat={sample.get('lat')}, lon={sample.get('lon')}, alt={sample.get('alt')}, "
        f"fix={sample.get('fix')}, quality={sample.get('fix_quality')}, "
        f"rtk_fixed={sample.get('rtk_fixed')}, speed_mps={sample.get('speed_mps')}, "
        f"speed_kph={sample.get('speed_kph')}, course_deg={sample.get('course_deg')}, "
        f"course_valid={sample.get('course_valid')}, vel_n_mps={sample.get('vel_n_mps')}, "
        f"vel_e_mps={sample.get('vel_e_mps')}, hdop={sample.get('hdop')}, "
        f"num_sv={sample.get('num_sv')}, diff_age={sample.get('diff_age')}"
    )

import sys

def main():
    print(sys.argv[1])
    gps = GPS(sys.argv[1], return_dict=True)
    print("Starting GPS() standalone test using class defaults.")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            print(_format_output(gps.run()))
            sleep(0.5)
    except KeyboardInterrupt:
        print("Stopping GPS test...")
    finally:
        gps.shutdown()


if __name__ == "__main__":
    main()
