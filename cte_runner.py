import donkeycar as dk

from parts.uart_backup import UART_backup_driver
from parts.gps import GPS
from parts.gps_to_xy import GPS_to_xy
from parts.health_check import HealthCheck
from parts.pure_pursuit_controller import PurePursuitController
from parts.threaded_socket_pub_part import ThreadedTelemetryStreamer
from parts.logger_gps import Logger_GPS
from parts.bno086 import BNO086
from parts.heading_fusion import HeadingFusion
from parts.loop_clock import LoopClock
# from parts.gstreamer_video_sync import GStreamerUvcRecorder

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

    V.add(LoopClock(), inputs=[], outputs=["loop/index", "loop/monotonic_ns", "loop/wall_time"])

    '''
    video_recorder = GStreamerUvcRecorder(
        device="/dev/video4",
        width=1280,
        height=720,
        fps=30,
        source_format="mjpeg",
    )
    V.add(
        video_recorder,
        inputs=["loop/index", "loop/monotonic_ns", "loop/wall_time"],
        outputs=[
            "video/nearest_camera_frame_id",
            "video/nearest_frame_pts_ns",
            "video/nearest_frame_monotonic_ns",
            "video/time_s",
            "video/delta_loop_to_frame_ms",
            "video/path",
        ],
        threaded=False,
    )
    '''

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

    # Pure Pursuit controller.
    # NOTE: this still expects the existing "_xy_throttle" path naming convention.
    csv_xy_path = args.file_name.split('.')[0] + "_xy_throttle" + ".csv"
    controller = PurePursuitController(
        path_csv=csv_xy_path,
        wheelbase_m=1.05,
        steer_max_deg=30.0,
        lookahead_time_s=0.6,
        min_lookahead_m=3.0,
        max_lookahead_m=12.0,
        search_window=80,
        max_resync_dist_m=15.0,
        rejoin_dist_m=2.0,
        rejoin_lookahead_m=3.0,
        rejoin_steer_gain=0.4,
        fallback_erpm=1500,
        throttle_floor_erpm=0,
        throttle_ceiling_erpm=3000,
        behind_target_erpm=1500,
        off_path_slowdown_m=1.2,
        off_path_stop_m=3.0,
        off_path_slowdown_erpm=1500,
        reverse_path=True,
        curvature_to_steering="empirical",
        steering_sign=1.0,
        verbose=True
    )

    V.add(
        controller,
        inputs=["fused_x", "fused_y", "fused_yaw", "gps_speed_mps"],
        outputs=["controls/throttle", "controls/steering", "pp/debug"],
        threaded=False,
    )

    V.add(ThreadedTelemetryStreamer(), inputs=['lat_raw','lon_raw','fused_yaw', 'controls/steering','imu_heading'])

    V.add(
        Logger_GPS(),
        inputs=[
            'lat_raw',
            'lon_raw',
            'controls/steering',
            'controls/throttle',
            'fix',
            'gps_heading',
            'gps_speed_mps',
            'imu_heading',
            'imu_accuracy_deg',
            'fused_x',
            'fused_y',
            'fused_yaw',
            'pp/debug',
            'loop/index',
            'loop/monotonic_ns',
            'loop/wall_time',
            'video/nearest_camera_frame_id',
            'video/nearest_frame_pts_ns',
            'video/nearest_frame_monotonic_ns',
            'video/time_s',
            'video/delta_loop_to_frame_ms',
            'video/path',
        ],
        outputs=[],
    )

    V.start(rate_hz=50, max_loop_count=None)
