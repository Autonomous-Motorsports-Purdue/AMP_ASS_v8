import donkeycar as dk

from parts.uart_backup import UART_backup_driver
from parts.gps import GPS
from parts.gps_to_xy import GPS_to_xy
from parts.health_check import HealthCheck
from parts.cte_controller import CTEController
from parts.threaded_socket_pub_part import ThreadedTelemetryStreamer
from parts.logger_gps import Logger_GPS
from parts.bno086 import BNO086
from parts.heading_fusion import HeadingFusion

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

    # data = np.genfromtxt(args.file_name, delimiter=',', skip_header=1, dtype=float, encoding='utf-8')
    data = np.genfromtxt(args.file_name, delimiter=',', skip_header=1, dtype=float, encoding='utf-8')
    
    ref_lat0 = data[0,0]
    ref_lon0 = data[0,1]

    V = dk.vehicle.Vehicle()

    # Heart beat
    heartbeat= HealthCheck("192.168.12.25", 6000) # just returns true rn
    V.add(heartbeat, inputs=[], outputs=["safety/heartbeat"])

    # # UART driver
    uart = UART_backup_driver("/dev/ttyACM2")
    V.add(uart, inputs=["controls/throttle", "controls/steering", "safety/heartbeat"], outputs=[], threaded=False)

    gps = GPS('/dev/ttyACM1')
    V.add(gps, inputs=[], outputs=['lat_raw', 'lon_raw', 'alt', 'fix', 'corr_age', 'hdop', 'sat_count', 'gps_heading', 'gps_speed_mps'], threaded=True)

    # IMU
    imu = BNO086(port='/dev/ttyACM0')
    V.add(imu, inputs=[], outputs=['imu_heading', 'imu_accuracy_deg', 'imu_lin_accel', 'imu_gyro'], threaded=True)

    # GPS to XY
    gps_to_xy = GPS_to_xy(ref_lat_deg=ref_lat0, ref_lon_deg=ref_lon0) # first point as origin
    V.add(gps_to_xy, inputs=["lat_raw", "lon_raw"], outputs=["x", "y", "gps_yaw"], threaded=False)

    heading_fusion = HeadingFusion()
    V.add(
        heading_fusion,
        inputs=["x", "y", "gps_yaw", "gps_speed_mps", "imu_heading", "imu_accuracy_deg", "imu_lin_accel", "imu_gyro"],
        outputs=["fused_x", "fused_y", "fused_yaw"],
        threaded=False,
    )

    # PID Controller
    # NOTE: must include "_throttle", with hardcoded throttle labels.
    csv_xy_path = args.file_name.split('.')[0] + "_xy_throttle" + ".csv"
    throttle = 2500
    kp, ki, kd = 0.3, 0.0, 0.3
    y_kp, y_ki, y_kd = 0.001, 0.0, 0.000001
    controller = CTEController(path_csv=csv_xy_path, throttle=throttle, kp=kp, ki=ki, kd=kd, kp_tangent=y_kp, ki_tangent=y_ki, kd_tangent=y_kd)

    V.add(controller, inputs=["fused_x", "fused_y", "fused_yaw"], outputs=["controls/throttle", "controls/steering"], threaded=False)

    V.add(ThreadedTelemetryStreamer(), inputs=['lat_raw','lon_raw','fused_yaw', 'controls/steering'])

    V.add(Logger_GPS(), inputs=['lat_raw','lon_raw', 'controls/steering', 'controls/throttle', 'fix', 'gps_heading', 'gps_speed_mps', 'imu_heading', 'imu_accuracy_deg', 'fused_x', 'fused_y', 'fused_yaw'], outputs=[])

    V.start(rate_hz=50, max_loop_count=None)
