import numpy as np
import time


class GPS_to_xy:
    """
    Convert geodetic coordinates (lat/lon in degrees) to a local Cartesian frame.

    Frame definition:
    - x: East (+x is to the right on a north-up plot)
    - y: North (+y is upward on a north-up plot)

    This uses a local tangent-plane approximation around a reference latitude/longitude.
    It is accurate for typical small- to medium-sized driving areas.
    """

    EARTH_RADIUS_M = 6378137.0  # WGS84 equatorial radius

    def __init__(self, ref_lat_deg: float, ref_lon_deg: float, min_heading_step_m: float = 0.3):
        self.ref_lat_deg = float(ref_lat_deg)
        self.ref_lon_deg = float(ref_lon_deg)
        self.ref_lat_rad = np.radians(self.ref_lat_deg)
        self.ref_lon_rad = np.radians(self.ref_lon_deg)
        self.cos_ref_lat = np.cos(self.ref_lat_rad)
        self.min_heading_step_m = float(min_heading_step_m)

        # Stateful heading estimate from consecutive local-frame displacements.
        self.prev_x_east_m = None
        self.prev_y_north_m = None
        self.gps_yaw_deg = None

    def _update_gps_yaw(self, x_east_m: float, y_north_m: float):
        """
        Update and return heading in degrees from +x (east), CCW positive.

        If displacement is too small, keep the previous yaw to suppress jitter.
        """
        if self.prev_x_east_m is None or self.prev_y_north_m is None:
            print('[GPS_TO_XY] Prev is None, cannot compute yaw')
            self.prev_x_east_m = x_east_m
            self.prev_y_north_m = y_north_m
            return self.gps_yaw_deg

        dx = x_east_m - self.prev_x_east_m
        dy = y_north_m - self.prev_y_north_m
        step_m = float(np.hypot(dx, dy))

        if step_m >= self.min_heading_step_m:
            self.gps_yaw_deg = float(np.degrees(np.arctan2(dy, dx)))

        self.prev_x_east_m = x_east_m
        self.prev_y_north_m = y_north_m
        return self.gps_yaw_deg

    def to_xy(self, lat_deg: float, lon_deg: float):
        """Return (x, y) in meters where x=east and y=north."""
        if lat_deg is None or lon_deg is None:
            print("GPS_TO_XY got NONE. Sleeping for 1s")
            time.sleep(1)
            return None, None
        lat_rad = np.radians(lat_deg)
        lon_rad = np.radians(lon_deg)

        dlat = lat_rad - self.ref_lat_rad
        dlon = lon_rad - self.ref_lon_rad

        x_east_m = dlon * self.cos_ref_lat * self.EARTH_RADIUS_M
        y_north_m = dlat * self.EARTH_RADIUS_M
        return x_east_m, y_north_m

    def to_latlon(self, x_east_m: float, y_north_m: float):
        """Inverse of to_xy: return (lat_deg, lon_deg)."""
        lat_rad = self.ref_lat_rad + (y_north_m / self.EARTH_RADIUS_M)
        lon_rad = self.ref_lon_rad + (x_east_m / (self.EARTH_RADIUS_M * self.cos_ref_lat))
        return float(np.degrees(lat_rad)), float(np.degrees(lon_rad))

    def run(self, lat_deg: float, lon_deg: float):
        """Run method for DonkeyCar part interface.

        Returns:
            (x_east_m, y_north_m, gps_yaw_deg)
        """
        print(f"lat_deg:{lat_deg},lon_deg:{lon_deg}")
        if lat_deg is None or lon_deg is None:
            print(f"[GPS_TO_XY]lat/lon none. ret 0")
            return 0,0,0
        x_east_m, y_north_m = self.to_xy(lat_deg, lon_deg)
        if x_east_m is None or y_north_m is None:
            return None, None, self.gps_yaw_deg

        gps_yaw_deg = self._update_gps_yaw(x_east_m, y_north_m)
        yaw_str = "None" if gps_yaw_deg is None else f"{gps_yaw_deg:.2f} deg"
        print(
            f"GPS_to_xy: lat {lat_deg:.6f}, lon {lon_deg:.6f} -> "
            f"x {x_east_m:.2f} m, y {y_north_m:.2f} m, yaw {yaw_str}"
        )
        return x_east_m, y_north_m, gps_yaw_deg


class XY_to_GPS:
    """Inverse of GPS_to_xy: local east/north meters -> lat/lon degrees."""

    EARTH_RADIUS_M = 6378137.0  # WGS84 equatorial radius

    def __init__(self, ref_lat_deg: float, ref_lon_deg: float):
        self.ref_lat_deg = float(ref_lat_deg)
        self.ref_lon_deg = float(ref_lon_deg)
        self.ref_lat_rad = np.radians(self.ref_lat_deg)
        self.ref_lon_rad = np.radians(self.ref_lon_deg)
        self.cos_ref_lat = np.cos(self.ref_lat_rad)

    def to_latlon(self, x_east_m: float, y_north_m: float):
        """Return (lat_deg, lon_deg) from local east/north meters."""
        if x_east_m is None or y_north_m is None:
            return None, None
        lat_rad = self.ref_lat_rad + (float(y_north_m) / self.EARTH_RADIUS_M)
        lon_rad = self.ref_lon_rad + (
            float(x_east_m) / (self.EARTH_RADIUS_M * self.cos_ref_lat)
        )
        return float(np.degrees(lat_rad)), float(np.degrees(lon_rad))

    def run(self, x, y):
        return self.to_latlon(x, y)