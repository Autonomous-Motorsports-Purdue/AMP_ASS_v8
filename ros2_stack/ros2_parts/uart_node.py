#!/usr/bin/env python3
"""ROS 2 port of parts/uart.py.

Drives the kart over the framed binary UART protocol. The kart moves only
while the heartbeat is true, so a heartbeat topic that has never been
published stops the kart rather than letting it run.

subscribes: cmd/throttle, cmd/steering (Float64, normalized -1 to 1),
            safety/heartbeat (Bool)
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float64
import serial

from ros2_parts.parameters import declare

START_BYTE = 254
END_BYTE = 255


class UartNode(Node):
    def __init__(self):
        super().__init__("uart")

        port_name = declare(self, "port_name", "/dev/ttyACM0", "Serial port the main PCB is on.")
        baudrate = declare(self, "baudrate", 115200, "Serial baud rate.")
        rate_hz = declare(self, "rate_hz", 30.0, "Rate commands are sent at, in Hz.")

        # configure the serial connections (the parameters differs on the device you are connecting to)
        self.ser = serial.Serial(port=port_name, baudrate=baudrate)
        self.get_logger().info(f"Connected to {port_name} @ {baudrate}")

        self.curr_v = 0
        self.curr_s = 0

        self.throttle = None
        self.steering = None
        self.alive = None

        self.create_subscription(Float64, "cmd/throttle", self.on_throttle, 10)
        self.create_subscription(Float64, "cmd/steering", self.on_steering, 10)
        self.create_subscription(Bool, "safety/heartbeat", self.on_heartbeat, 10)
        self.create_timer(1.0 / rate_hz, self.send_command)

    def on_throttle(self, msg):
        self.throttle = msg.data

    def on_steering(self, msg):
        self.steering = msg.data

    def on_heartbeat(self, msg):
        self.alive = msg.data

    def update_velocity(self, new_v):
        # shifting values into UART accepted range (128-255) (zero at 191)
        if new_v < 0:
            new_v = 0
        elif new_v > 255:
            new_v = 255
        new_v = new_v >> 1

        self.curr_v = new_v

    def update_steering(self, new_s):
        # shifting values into UART accepted range (128-255) (zero at 191)
        if new_s <= -63:
            new_s = 0
        elif new_s >= 64:
            new_s = 127
        else:
            new_s = new_s + 64

        self.curr_s = new_s

    def reset_kart(self):
        self.update_velocity(0)
        self.update_steering(0)
        self.write_serial()

    def write_serial(self):
        # send start byte
        self.ser.write(START_BYTE.to_bytes(1, "little"))

        # send current velocity and steering
        self.ser.write(self.curr_v.to_bytes(1, "little"))
        self.ser.write(self.curr_s.to_bytes(1, "little"))

        # send end byte
        self.ser.write(END_BYTE.to_bytes(1, "little"))

    def run(self, v, s, alive):
        """
        Steering and throttle both come in on (-1, 1).
        """
        if not alive:
            self.reset_kart()
            return
        v = int(v * 127)  # throttle from -127 to 127

        # steering is centered at 128
        s = int(s * 127)

        # clip throttle to (-100, 100)
        v = max(-100, min(100, v))

        self.get_logger().debug(f"Throttle: {v}, Steering: {s}")

        self.update_velocity(v)
        self.update_steering(s)
        self.write_serial()

    def send_command(self):
        if not self.alive:
            self.get_logger().warn("no heartbeat, holding the kart", throttle_duration_sec=5.0)
        v = self.throttle if self.throttle is not None else 0.0
        s = self.steering if self.steering is not None else 0.0
        self.run(v, s, self.alive)

    def destroy_node(self):
        if self.ser is not None and self.ser.is_open:
            self.reset_kart()
            self.ser.close()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = UartNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
