import threading

import donkeycar as dk

from parts.uart_backup import UART_backup_driver
from parts.health_check import HealthCheck

RATE_HZ = 50


class ManualSteering:
    """Read steering from stdin in a background thread; UART loop keeps running."""

    def __init__(self):
        self.steer = 0.0
        self._lock = threading.Lock()
        threading.Thread(target=self._input_loop, daemon=True).start()

    def _input_loop(self):
        print("Enter steer (-1 to 1), press Enter to update:")
        while True:
            try:
                val = float(input("> "))
                val = max(-1.0, min(1.0, val))
                with self._lock:
                    self.steer = val
                print(f"  -> steering {val}  (UART {int(127.5 * (val + 1))})")
            except ValueError:
                print("  invalid — enter a number between -1 and 1")

    def run(self):
        with self._lock:
            return self.steer, 0, "RTK FIXED"


if __name__ == "__main__":
    V = dk.vehicle.Vehicle()

    heartbeat = HealthCheck("192.168.12.25", 6000)
    V.add(heartbeat, inputs=[], outputs=["safety/heartbeat"])

    V.add(ManualSteering(), inputs=[], outputs=["controls/steering", "controls/throttle", "fix"], threaded=False)

    uart = UART_backup_driver("/dev/ttyACM2", quiet=True)
    V.add(
        uart,
        inputs=["controls/throttle", "controls/steering", "safety/heartbeat", "fix"],
        outputs=["steer_actual", "throttle_actual"],
        threaded=False,
    )

    print(f"Manual steering test at {RATE_HZ} Hz — type a value and press Enter, Ctrl+C to stop")
    V.start(rate_hz=RATE_HZ)
