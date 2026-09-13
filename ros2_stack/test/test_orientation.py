"""Unit tests for the angle and quaternion helpers.

These need a sourced ROS 2 environment for ``geometry_msgs``; run them with
``colcon test --packages-select ros2_parts``.
"""

import math

import pytest

from ros2_parts.orientation import (
    compass_heading_to_enu_yaw_deg,
    enu_yaw_to_compass_heading_deg,
    normalize_angle_deg,
    normalize_angle_rad,
    quaternion_to_euler_deg,
    quaternion_to_yaw,
    normalize_quaternion,
    wrap360,
    yaw_to_quaternion,
)


class TestYawQuaternion:
    """Round-tripping a yaw through a quaternion."""

    def test_zero_yaw_is_the_identity_rotation(self) -> None:
        quaternion = yaw_to_quaternion(0.0)
        assert (quaternion.x, quaternion.y, quaternion.z, quaternion.w) == (
            pytest.approx(0.0), pytest.approx(0.0), pytest.approx(0.0), pytest.approx(1.0))

    def test_quarter_turn_about_z(self) -> None:
        quaternion = yaw_to_quaternion(math.pi / 2.0)
        assert quaternion.z == pytest.approx(math.sqrt(0.5))
        assert quaternion.w == pytest.approx(math.sqrt(0.5))

    @pytest.mark.parametrize('yaw_rad', [0.0, 0.5, -0.5, 1.5, -3.0, math.pi - 1e-6])
    def test_round_trip(self, yaw_rad: float) -> None:
        assert quaternion_to_yaw(yaw_to_quaternion(yaw_rad)) == pytest.approx(yaw_rad)

    def test_euler_agrees_with_the_yaw_helper(self) -> None:
        quaternion = yaw_to_quaternion(math.radians(30.0))
        _roll, _pitch, yaw = quaternion_to_euler_deg(
            quaternion.x, quaternion.y, quaternion.z, quaternion.w)
        assert yaw == pytest.approx(30.0)


class TestNormalizeQuaternion:
    """Rejecting quaternions that are not rotations."""

    def test_scales_to_unit_length(self) -> None:
        result = normalize_quaternion(0.0, 0.0, 2.0, 0.0)
        assert result == pytest.approx((0.0, 0.0, 1.0, 0.0))

    @pytest.mark.parametrize(
        'components',
        [(0.0, 0.0, 0.0, 0.0), (10.0, 0.0, 0.0, 0.0), (math.nan, 0.0, 0.0, 1.0)],
    )
    def test_rejects_anything_far_from_unit_length(self, components) -> None:
        assert normalize_quaternion(*components) is None


class TestAngleWrapping:
    """Wrapping angles into their canonical ranges."""

    @pytest.mark.parametrize(
        'angle, expected', [(0.0, 0.0), (370.0, 10.0), (-10.0, 350.0), (360.0, 0.0)])
    def test_wrap360(self, angle: float, expected: float) -> None:
        assert wrap360(angle) == pytest.approx(expected)

    @pytest.mark.parametrize(
        'angle, expected', [(0.0, 0.0), (190.0, -170.0), (-190.0, 170.0), (180.0, -180.0)])
    def test_normalize_angle_deg(self, angle: float, expected: float) -> None:
        assert normalize_angle_deg(angle) == pytest.approx(expected)

    def test_normalize_angle_rad(self) -> None:
        # The range is half-open, so a half turn lands on -pi.
        assert normalize_angle_rad(3.0 * math.pi) == pytest.approx(-math.pi)
        assert normalize_angle_rad(-1.5 * math.pi) == pytest.approx(math.pi / 2.0)


class TestCompassConversion:
    """Compass headings against ENU yaw."""

    @pytest.mark.parametrize(
        'heading_deg, yaw_deg',
        [(0.0, 90.0), (90.0, 0.0), (180.0, -90.0), (270.0, -180.0)],
    )
    def test_cardinal_directions(self, heading_deg: float, yaw_deg: float) -> None:
        # North is 0 on a compass but 90 in ENU yaw; east is 90 but 0.
        assert compass_heading_to_enu_yaw_deg(heading_deg) == pytest.approx(yaw_deg)

    @pytest.mark.parametrize('heading_deg', [0.0, 45.0, 123.4, 270.0, 359.9])
    def test_round_trip(self, heading_deg: float) -> None:
        yaw = compass_heading_to_enu_yaw_deg(heading_deg)
        assert enu_yaw_to_compass_heading_deg(yaw) == pytest.approx(heading_deg)
