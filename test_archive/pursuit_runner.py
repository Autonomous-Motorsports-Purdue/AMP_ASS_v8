import donkeycar as dk
from donkeycar.parts.controller import LocalWebController

import pickle
import time
from parts.health_check import HealthCheck
from parts.uart import UART_Driver
from parts.uart_backup import UART_backup_driver
from constants import DRIVE_LOOP_HZ
from parts.logger import Logger
from parts.frame_publisher import Frame_Publisher
from parts.segment_model import Segment_Model
from parts.curve_fit import Curve_fit
from parts.translate import Translate
from parts.pure_pursuit import Pure_Pursuit
from parts.control_mux import Control_Muxer

if __name__ == "__main__":
    # web controller
    V = dk.vehicle.Vehicle()
    with open("zed_calibration_params.bin", "rb") as f:
        b = pickle.load(f)
    
    translate = Translate(b, 0.99695, -5.25)
    heartbeat= HealthCheck("192.168.12.25", 6000) # aryamaan has ip 100
    V.add(heartbeat, inputs=[], outputs=["safety/heartbeat"])
    V.add(Frame_Publisher(), outputs=['sensors/ZED/RGB/left', 'sensors/ZED/RGB/right'], threaded=False)
    V.add(Segment_Model(), inputs=['sensors/ZED/RGB/left'], outputs=['lane', 'drive'])
    #V.add(LaneDetect(), inputs=['sensors/ZED/RGB/left', ' ', ' '], outputs=['points', 'perception/segmentedTrack'], threaded=False)
    V.add(Curve_fit(), inputs=['lane', 'drive'], outputs=['waypoint', 'lines'])
    V.add(translate, inputs=['waypoint'], outputs=['world'])

    V.add(Pure_Pursuit(0.3), inputs=["world"], outputs=["controls/auto_steering","controls/auto_throttle"])
    
    
    controller = LocalWebController()
    # web controller just expects all these things even though they don't exist
    V.add(
        controller,
        inputs=["lines", "tub/num_records", "user/mode", "recording"],
        outputs=[
            "user/steering_fake",
            "user/throttle_fake",
            "user/mode",
            "recording",
            "web/buttons",
        ],
        threaded=True,
    )

    # control muxer
    control_mux = Control_Muxer()
    V.add(control_mux, inputs=['user_steering_fake', 'user/throttle_fake', 'controls/auto_steering', 'controls/auto_throttle'], outputs=['controls/steering', 'controls/throttle'])
    # uart controller
    V.add(Logger(), inputs=['sensors/ZED/RGB/left', 'lines', 'waypoint', 'controls/steering', 'controls/throttle'], threaded=False)
    uart = UART_backup_driver("/dev/ttyACM0")
    V.add(
        uart,
        inputs=["controls/throttle", "controls/steering", "safety/heartbeat"],
        outputs=[],
        threaded=False,
    )

    V.start(rate_hz=DRIVE_LOOP_HZ)
