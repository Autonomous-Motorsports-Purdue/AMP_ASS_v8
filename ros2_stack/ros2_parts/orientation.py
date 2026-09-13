"""Angle and quaternion conversions shared by the nodes.

ROS 2 orientation is an ENU quaternion: yaw counter-clockwise from east, in
radians. Several of these sensors and controllers speak compass degrees, so
the conversions live here instead of in every node. The quaternion helpers
are the ones from parts/bno086.py.
"""

import math

from geometry_msgs.msg import Quaternion


def yaw_to_quaternion(yaw_rad):
    half_yaw = 0.5 * yaw_rad
    return Quaternion(x=0.0, y=0.0, z=math.sin(half_yaw), w=math.cos(half_yaw))


def quaternion_to_yaw(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def quaternion_to_euler_deg(x, y, z, w):
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


def normalize_quaternion(x, y, z, w):
    """Scale to unit length, or return None if this is not a rotation."""
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if not math.isfinite(norm) or not 0.5 < norm < 1.5:
        return None
    return x / norm, y / norm, z / norm, w / norm


def wrap360(angle_deg):
    angle_deg %= 360.0
    if angle_deg < 0.0:
        angle_deg += 360.0
    return angle_deg


def normalize_angle_deg(angle_deg):
    return (float(angle_deg) + 180.0) % 360.0 - 180.0


def normalize_angle_rad(angle_rad):
    return (float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi


def compass_heading_to_enu_yaw_deg(heading_deg):
    """Compass runs clockwise from north, ENU yaw counter-clockwise from east."""
    return normalize_angle_deg(90.0 - float(heading_deg))


def enu_yaw_to_compass_heading_deg(yaw_deg):
    return wrap360(90.0 - float(yaw_deg))
