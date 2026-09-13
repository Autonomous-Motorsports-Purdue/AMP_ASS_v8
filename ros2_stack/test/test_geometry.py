"""Unit tests for the path geometry.

The sign convention here decides which way the kart steers, so it is pinned
down explicitly rather than left to the caller to infer.
"""

import math

import pytest

from ros2_parts.geometry import cross_track_error, distance


class TestDistance:
    """Straight-line distance between two points."""

    def test_known_triangle(self) -> None:
        assert distance(0.0, 0.0, 3.0, 4.0) == pytest.approx(5.0)

    def test_zero_for_the_same_point(self) -> None:
        assert distance(1.5, -2.5, 1.5, -2.5) == pytest.approx(0.0)


class TestCrossTrackError:
    """Signed distance from a path segment to a point."""

    def test_left_of_the_path_is_positive(self) -> None:
        # Travelling east, north is to the left.
        assert cross_track_error((0, 0), (10, 0), (5, 2)) == pytest.approx(2.0)

    def test_right_of_the_path_is_negative(self) -> None:
        assert cross_track_error((0, 0), (10, 0), (5, -2)) == pytest.approx(-2.0)

    def test_on_the_path_is_zero(self) -> None:
        assert cross_track_error((0, 0), (10, 0), (5, 0)) == pytest.approx(0.0)

    def test_the_convention_holds_in_every_direction(self) -> None:
        # Travelling north, west is to the left.
        assert cross_track_error((0, 0), (0, 10), (-3, 5)) == pytest.approx(3.0)
        assert cross_track_error((0, 0), (0, 10), (3, 5)) == pytest.approx(-3.0)
        # Travelling west, south is to the left.
        assert cross_track_error((0, 0), (-10, 0), (-5, -2)) == pytest.approx(2.0)

    def test_magnitude_is_perpendicular_not_along_track(self) -> None:
        # The point is 4 m off a diagonal path, measured perpendicular.
        error = cross_track_error((0, 0), (10, 10), (0, 4))
        assert abs(error) == pytest.approx(4.0 / math.sqrt(2.0))
        assert error > 0.0

    def test_the_point_may_lie_beyond_the_segment(self) -> None:
        # The segment defines an infinite line; the point need not be on it.
        assert cross_track_error((0, 0), (1, 0), (50, 3)) == pytest.approx(3.0)

    def test_a_degenerate_segment_has_no_direction(self) -> None:
        assert cross_track_error((1, 1), (1, 1), (5, 5)) == pytest.approx(0.0)

    def test_reversing_the_segment_flips_the_sign(self) -> None:
        forward = cross_track_error((0, 0), (10, 0), (5, 2))
        backward = cross_track_error((10, 0), (0, 0), (5, 2))
        assert forward == pytest.approx(-backward)
