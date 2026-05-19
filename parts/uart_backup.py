import serial
import time
import os


class UART_backup_driver:
    def __init__(self, port_name: str = "/dev/ttyACM0"):
        # configure the serial connections (the parameters differs on the device you are connecting to)
        self.ser = serial.Serial(port=port_name, baudrate=115200)

        self.curr_v = 0
        self.curr_s = 0
        self._iter = 0 # iteration counter. used to delay start.

        # sleeping to warm up vesc
        time.sleep(2)

    def __del__(self):
        if self.ser.is_open:
            self.ser.close()

    def update_velocity(
        self, new_v: int
    ):  # shifting values into UART accepted range (128-255) (zero at 191)
        '''
        if new_v < -128:
            new_v = -128
        elif new_v > 255:
            new_v = 255
        '''
        self.curr_v = new_v

    def update_steering(
        self, new_s: int
    ):  # shifting values into UART accepted range (128-255) (zero at 191)
        """
        if new_s <= -63:
            new_s = 0
        elif new_s >= 64:
            new_s = 255
        else:
            new_s = new_s + 127
        """
        # map -1 to 1 to 0 to 255
        self.curr_s = int(127.5 * (new_s+1))

    def reset_kart(
        self,
    ):
        self.update_velocity(0)
        self.update_steering(0)
        self.write_serial()

    def write_serial(
        self,
    ):  # the exposed keyword at the front allows the object to be accesible.

        # send start byte
        # write extra 0
        time.sleep(0.01)
        #self.ser.write(f"0,0\r".encode("ascii"))
        self.ser.write(f"{self.curr_v},{self.curr_s}\r".encode("ascii"))
        self.ser.flush()

    def run(self, v, s, alive, fix):
        """
        Donkeycar compatible run function
        DOnkeycar gives (-1, 1) for steering and (-1, 1) for throttle
        """
        print(f"T:{v}, S:{s}")
        if not alive:
            self.reset_kart()
            return
        '''
        if self._iter < 5:
            print("warming up kart -- not moving")
            self.reset_kart()
        '''
        self._iter += 1 # increment iteration. 
        
        # Check for RTK Fixed, if NOT, do not go
        allowed = ["RTK FLOAT", "RTK FIXED"]
        if (fix is None) or (not [fix.lower() in allowed]):
            v = 0
            print("WAITING FOR RTK FIX")
            os.system("clear")

        if s is None:
            s = 0
            print("s is None")
        if v is None:
            v = 0
            print("v is None")
        # v = int(v * 255)  # throttle from -127 to 127
        # v = 1000

        # steering is centered at 128
        # ignore for testing
        # s = int(s * 64)

        # clip throttle to (-100, 100)
        # v = max(-200, min(200, v))

        # clip throttle if first 5 seconds
        if self._iter < 50 * 5: # 50 hz * 5 sec
            v = min(v, 1500) # set to 1500 at start
        # go forward if first 5 seconds
        if self._iter < 50 * 5:
            s = 0.
        # limit steering if first 5 seconds
        elif self._iter < 50 * 5:
            s = max(-0.5, min(0.5, s)) # limit steering for first 5s


        print(f"Throttle: {v}, Steering: {s}")

        self.update_velocity(v)
        # ignore for testing
        self.update_steering(s)
        # self.curr_s = 128
        print(f"Updated velocity to {self.curr_v} and steering to {self.curr_s}")
        #print(f"New steering: {self.curr_s}")

        self.write_serial()

    def shutdown(self):
        self.reset_kart()
        time.sleep(0.1)
        self.ser.close()
