#!/usr/bin/env python3

import math
import re
import time


KV_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^,\s]+)")


def wrap360(angle):
    angle %= 360.0
    if angle < 0.0:
        angle += 361.0
    return angle


def parse_float(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def parse_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def field_float(fields, name, default=None):
    value = parse_float(fields.get(name))
    return default if value is None else value


def field_int(fields, name, default=None):
    value = parse_int(fields.get(name))
    return default if value is None else value


def normalize_quat(x, y, z, w):
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if not math.isfinite(norm) or not 0.5 < norm < 1.5:
        return None
    return x / norm, y / norm, z / norm, w / norm


def quat_to_euler_deg(x, y, z, w):
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def math_yaw_to_compass_heading(math_yaw_deg):
    return wrap360(90.0 - math_yaw_deg)


def direction_label(heading_deg):
    labels = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
    return labels[int((wrap360(heading_deg) + 22.5) // 45.0) % len(labels)]


def compass_text(heading_deg):
    direction = direction_label(heading_deg)
    return " ".join(f"[{label}]" if label == direction else label for label in ("N", "NE", "E", "SE", "S", "SW", "W", "NW"))


class BNO086:
    '''
    bno086
    '''
    def __init__(
        self,
        port="/dev/ttyACM0",
        baudrate=460800,
        timeout=0.05,
        declination=-4.55,
        mount_offset=0.0,
        invert=False,
        poll_delay=0.0025,
        donkey=True
    ):
        import serial

        self.ser = serial.Serial(port, baudrate, timeout=timeout)
        time.sleep(2.0)

        self.declination = declination
        self.mount_offset = mount_offset
        self.invert = invert
        self.poll_delay = poll_delay
        self.running = True
        self.latest = None
        self.donkey = donkey

    def run(self):
        while self.ser.in_waiting:
            line = self.ser.readline().decode(errors="ignore").strip()
            sample = self._parse_line(line)
            if sample is not None:
                self.latest = sample

        return self.latest

    def update(self):
        while self.running:
            self.run()
            time.sleep(self.poll_delay)

    def run_threaded(self):
        return self.latest

    def shutdown(self):
        self.running = False
        if self.ser and self.ser.is_open:
            self.ser.close()

    def _parse_line(self, line):
        fields = dict(KV_RE.findall(line))
        if not {"qi", "qj", "qk", "qr"}.issubset(fields):
            return None

        qi = parse_float(fields["qi"])
        qj = parse_float(fields["qj"])
        qk = parse_float(fields["qk"])
        qr = parse_float(fields["qr"])
        if None in (qi, qj, qk, qr):
            return None

        quat = normalize_quat(qi, qj, qk, qr)
        if quat is None:
            return None

        x, y, z, w = quat
        if self.invert:
            x, y, z = -x, -y, -z

        roll, pitch, yaw = quat_to_euler_deg(x, y, z, w)
        heading = wrap360(math_yaw_to_compass_heading(yaw) + self.mount_offset + self.declination)

        if not self.donkey:
            return {
                "heading": heading,
                "roll": roll,
                "pitch": pitch,
                "yaw": wrap360(yaw),
                "quat": (x, y, z, w),
                "accuracy_deg": field_float(fields, "accuracy_deg"),
                "lin_accel": (
                    field_float(fields, "lin_ax_mps2", math.nan),
                    field_float(fields, "lin_ay_mps2", math.nan),
                    field_float(fields, "lin_az_mps2", math.nan),
                ),
                "gyro_dps": (
                    field_float(fields, "gx_dps", math.nan),
                    field_float(fields, "gy_dps", math.nan),
                    field_float(fields, "gz_dps", math.nan),
                ),
                "seq": field_int(fields, "seq"),
                "t_ms": field_int(fields, "t_ms"),
                "raw": line,
            }

        if self.donkey: # NOTE: must return list
            return [
                heading, # HEADING
                field_float(fields, "accuracy_deg"), # ACCURACY
                ( # LIN_ACCEL
                    field_float(fields, "lin_ax_mps2", math.nan),
                    field_float(fields, "lin_ay_mps2", math.nan),
                    field_float(fields, "lin_az_mps2", math.nan),
                ),
                ( # GYRO
                    field_float(fields, "gx_dps", math.nan),
                    field_float(fields, "gy_dps", math.nan),
                    field_float(fields, "gz_dps", math.nan),
                ), 
            ]



def main():
    import argparse

    parser = argparse.ArgumentParser(description="Print BNO085/BNO086 car compass orientation")
    parser.add_argument("--port", default="/dev/ttyACM0", help="Serial port, e.g. /dev/ttyACM0")
    parser.add_argument("--baudrate", type=int, default=460800)
    parser.add_argument("--declination", type=float, default=-4.55)
    parser.add_argument("--mount-offset", type=float, default=0.0)
    parser.add_argument("--invert", action="store_true")
    args = parser.parse_args()

    imu = BNO086(
        port=args.port,
        baudrate=args.baudrate,
        declination=args.declination,
        mount_offset=args.mount_offset,
        invert=args.invert,
        donkey=False,
    )

    try:
        while True:
            sample = imu.run()
            if sample is not None:
                heading = sample["heading"]
                accuracy = sample["accuracy_deg"]
                accuracy_text = "unknown" if accuracy is None else f"{accuracy:.1f} deg"
                print(
                    f"\r{compass_text(heading)}  "
                    f"heading={heading:6.1f} deg {direction_label(heading):>2}  "
                    f"accuracy={accuracy_text:>9}  "
                    f"roll={sample['roll']:6.1f}  pitch={sample['pitch']:6.1f}",
                    end="",
                    flush=True,
                )
            time.sleep(0.02)
    except KeyboardInterrupt:
        print()
    finally:
        imu.shutdown()


if __name__ == "__main__":
    main()
