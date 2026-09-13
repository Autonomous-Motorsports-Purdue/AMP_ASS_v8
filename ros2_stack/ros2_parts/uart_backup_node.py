#!/usr/bin/env python3
"""ROS 2 port of parts/uart_backup.py.

Drives the kart over the ASCII UART protocol, gated on an RTK fix. Anything
missing counts as unsafe, so a heartbeat or fix topic that has never been
published stops the kart rather than letting it run.

subscribes: cmd/throttle, cmd/steering (Float64), safety/heartbeat (Bool),
            gps/fix_type (String)
publishes:  cmd/commanded_steering, cmd/commanded_throttle (Float64) --
            what was actually sent to the kart
"""

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float64, String
import serial

from ros2_parts.parameters import declare


class UartBackupNode(Node):
    def __init__(self):
        super().__init__("uart_backup")

        port_name = declare(self, "port_name", "/dev/ttyACM1", "Serial port the main PCB is on.")
        baudrate = declare(self, "baudrate", 115200, "Serial baud rate.")
        rate_hz = declare(self, "rate_hz", 50.0, "Rate commands are sent at, in Hz.")
        self.startup_iters = declare(
            self, "startup_cycles", 250,
            "Cycles at startup with the throttle clamped and the steering centered.")
        self.startup_throttle = declare(
            self, "startup_throttle", 1500.0, "Throttle cap during those cycles.")

        # configure the serial connections (the parameters differs on the device you are connecting to)
        self.ser = serial.Serial(port=port_name, baudrate=baudrate)
        self.get_logger().info(f"Connected to {port_name} @ {baudrate}")

        self.curr_v = 0
        self.curr_s = 0
        self._iter = 0 # iteration counter. used to delay start.

        # Steer rate limiting
        self.MAX_STEER_DIFF = declare(
            self, "max_steer_diff", 0.1, "Largest steering change per cycle.")
        self.prev_steer = 0

        self.throttle = None
        self.steering = None
        self.alive = None
        self.fix = None

        self.commanded_steer_pub = self.create_publisher(Float64, "cmd/commanded_steering", 10)
        self.commanded_throt_pub = self.create_publisher(Float64, "cmd/commanded_throttle", 10)

        self.create_subscription(Float64, "cmd/throttle", self.on_throttle, 10)
        self.create_subscription(Float64, "cmd/steering", self.on_steering, 10)
        self.create_subscription(Bool, "safety/heartbeat", self.on_heartbeat, 10)
        self.create_subscription(String, "gps/fix_type", self.on_fix_type, 10)
        self.create_timer(1.0 / rate_hz, self.send_command)

    def on_throttle(self, msg):
        self.throttle = msg.data

    def on_steering(self, msg):
        self.steering = msg.data

    def on_heartbeat(self, msg):
        self.alive = msg.data

    def on_fix_type(self, msg):
        self.fix = msg.data

    def update_velocity(self, new_v):
        self.curr_v = new_v

    def update_steering(self, new_s):
        # map -1 to 1 to 0 to 255
        self.curr_s = int(127.5 * (new_s+1))

    def reset_kart(self):
        self.update_velocity(0)
        self.update_steering(0)
        self.write_serial()

    def write_serial(self):
        self.ser.write(f"{self.curr_v},{self.curr_s}\r".encode("ascii"))
        self.ser.flush()

    def run(self, v, s, alive, fix):
        """
        Steering comes in on (-1, 1), throttle in eRPM.
        """
        if not alive:
            self.get_logger().warn("no heartbeat, holding the kart", throttle_duration_sec=5.0)
            self.reset_kart()
            return

        self._iter += 1 # increment iteration.

        # Check for RTK Fixed, if NOT, do not go
        allowed = ["RTK FLOAT", "RTK FIXED"]
        if (fix is None) or str(fix).strip() not in allowed:
            v = 0
            self.get_logger().warn("WAITING FOR RTK FIX", throttle_duration_sec=2.0)

        if s is None:
            s = 0
        if v is None:
            v = 0

        # clip throttle and go straight for the first few seconds
        if self._iter < self.startup_iters:
            v = min(v, self.startup_throttle) # set to 1500 at start
            s = 0.

        # slew rate
        s = np.clip(s, self.prev_steer - self.MAX_STEER_DIFF, self.prev_steer + self.MAX_STEER_DIFF)
        self.prev_steer = s

        self.get_logger().debug(f"Throttle: {v}, Steering: {s}")

        self.update_velocity(int(v))
        self.update_steering(s)
        self.write_serial()

        return s, v

    def send_command(self):
        commanded = self.run(self.throttle, self.steering, self.alive, self.fix)
        if commanded is None:
            return
        s, v = commanded
        self.commanded_steer_pub.publish(Float64(data=float(s)))
        self.commanded_throt_pub.publish(Float64(data=float(v)))

    def destroy_node(self):
        if self.ser is not None and self.ser.is_open:
            self.reset_kart()
            self.ser.close()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = UartBackupNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
