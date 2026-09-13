"""Path geometry for the cross-track controller.

Replaces the donkeycar.la helpers (la.Line3D, la.Vec3, utils.dist) the
cte_controller part imported, so nothing here depends on donkeycar.

The sign matters. The error goes straight into a PID whose output is a
steering command, so an inverted sign steers the wrong way.
"""

import math


def distance(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)


def cross_track_error(segment_start, segment_end, point):
    """Signed perpendicular distance from the segment's line to point.

    Positive when the point is left of travel from segment_start to
    segment_end, which is what the PID expects: left of the path gives a
    positive error and a rightward correction. A degenerate segment has no
    direction to measure against, so it returns 0.
    """
    start_x, start_y = float(segment_start[0]), float(segment_start[1])
    end_x, end_y = float(segment_end[0]), float(segment_end[1])
    point_x, point_y = float(point[0]), float(point[1])

    dir_x = end_x - start_x
    dir_y = end_y - start_y
    length = math.hypot(dir_x, dir_y)
    if length < 1e-9:
        return 0.0
    dir_x /= length
    dir_y /= length

    # split the offset from the segment start into along-path and across-path
    offset_x = point_x - start_x
    offset_y = point_y - start_y
    along = offset_x * dir_x + offset_y * dir_y
    error_x = offset_x - along * dir_x
    error_y = offset_y - along * dir_y

    # 2D cross product is positive when the point is left of the path
    cross = dir_x * error_y - dir_y * error_x
    sign = -1.0 if cross < 0.0 else 1.0
    return math.hypot(error_x, error_y) * sign
