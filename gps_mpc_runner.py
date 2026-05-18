import donkeycar as dk

from parts.gps_pid import SimpleGPSPIDController
from parts.uart_backup import UART_backup_driver
from parts.imu import IMU
from parts.gps import GPS
from parts.gps_to_xy import GPS_to_xy
from parts.health_check import HealthCheck

# from parts.gps_visualizer import GPSVisualizer

import numpy as np

'''
checklist if not working

1. verify IMU, GPS, Main PCB UART are plugged in.
2. verify ports for each. should be /dev/tty* or /dev/usb*. unplug to test

run this to download the offline map of WL: 
```
python 
```
'''
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_name", help="gps waypoint csv file name")
    args = parser.parse_args()

    data = np.genfromtxt(args.file_name, delimiter=',', skip_header=1, dtype=float, encoding='utf-8')
    ref_lat0 = data[0,0]
    ref_lon0 = data[0,1]

    V = dk.vehicle.Vehicle()

    # Heart beat
    heartbeat= HealthCheck("192.168.12.25", 6000) # just returns true rn
    V.add(heartbeat, inputs=[], outputs=["safety/heartbeat"])

    # # UART driver
    uart = UART_backup_driver("/dev/ttyACM0")
    V.add(uart, inputs=["controls/throttle", "controls/steering", "safety/heartbeat"], outputs=[], threaded=False)

    # IMU
    imu = IMU("/dev/ttyACM1")
    V.add(imu, inputs=[], outputs=["yaw_rate", "yaw", "a_x", "a_y"], threaded=False) # TODO: make this threaded

    gps = GPS('/dev/ttyACM2')
    V.add(gps, inputs=[], outputs=['lat_raw', 'lon_raw', 'alt', 'fix', 'corr_age', 'hdop', 'sat_count', 'gps_heading'], threaded=True)

    # GPS to XY
    gps_to_xy = GPS_to_xy(ref_lat_deg=ref_lat0, ref_lon_deg=ref_lon0) # first point as origin
    V.add(gps_to_xy, inputs=["lat_raw", "lon_raw"], outputs=["x", "y", "gps_yaw"], threaded=False)

    # MPC Controller
    csv_xy_path = args.file_name.split('.')[0] + "_xy" + ".csv"
    controller = SimpleGPSPIDController(
        path_csv=csv_xy_path,
        lookahead_m=args.lookahead_m,
        throttle=args.throttle,
        kp=args.kp,
        ki=args.ki,
        kd=args.kd,
    )
    V.add(
        controller,
        inputs=["lat_raw", "lon_raw", "yaw"],
        outputs=["controls/steering", "controls/throttle"],
        threaded=False,
    )

    V.start(rate_hz=100, max_loop_count=None) # Remove this once we do more.
