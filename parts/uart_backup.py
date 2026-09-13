import time

import serial


class UART_backup_driver:
    """Sends "throttle,steer\\r" to the main PCB. Steer -1..1 is mapped to 0..255."""

    def __init__(self, port_name: str = "/dev/ttyACM0"):
        self.ser = serial.Serial(port=port_name, baudrate=115200)
        time.sleep(2)  # vesc warm-up

    def send(self, v, s):
        time.sleep(0.01)  # REQUIRED: the PCB drops commands without this pause. Do not remove.
        self.ser.write(f"{v},{int(127.5 * (s + 1))}\r".encode("ascii"))
        self.ser.flush()

    def run(self, v, s, alive, fix):
        """Donkeycar run: throttle in raw counts, steering in (-1, 1)."""
        if not alive or str(fix).strip() not in ("RTK FLOAT", "RTK FIXED"):
            v = 0
        v, s = v or 0, s or 0.0
        self.send(v, s)
        return s, v

    def shutdown(self):
        self.send(0, 0.0)
        time.sleep(0.1)
        self.ser.close()
