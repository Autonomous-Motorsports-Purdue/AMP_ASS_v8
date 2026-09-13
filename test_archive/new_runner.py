import donkeycar as dk
from donkeycar.parts.controller import LocalWebController

import time
from parts.uart_backup import UART_backup_driver
from parts.health_check import HealthCheck
from constants import DRIVE_LOOP_HZ
from parts.pure_pursuit import Pure_Pursuit
from parts.test import Test

if __name__ == "__main__":
    # web controller
    V = dk.vehicle.Vehicle()
    heartbeat= HealthCheck("192.168.1.100", 6000) # aryamaan has ip 100
    V.add(heartbeat, inputs=[], outputs=["safety/heartbeat"])
    V.add(Test(), inputs=[], outputs=["user/throttle_fake","user/steering_fake"])
    """    
    controller = LocalWebController()
    # web controller just expects all these things even though they don't exist
    
    V.add(
        controller,
        inputs=["perception/segmentedTrack", "tub/num_records", "user/mode", "recording"],
        outputs=[
            "user/steering_fake",
            "user/throttle_fake",
            "user/mode",
            "recording",
            "web/buttons",
        ],
        threaded=True,
    )
    """
    # uart controller
    uart = UART_backup_driver("/dev/ttyACM0")
    V.add(
        uart,
        inputs=["user/throttle_fake", "user/steering_fake", "safety/heartbeat"],
        outputs=[],
        threaded=False,
    )

    V.start(rate_hz=DRIVE_LOOP_HZ)
