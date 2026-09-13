#!/usr/bin/env python3
"""ROS 2 port of parts/uart_backup2.py.

Bench rig that writes a raw throttle byte straight to the VESC. This is a test
fixture, not part of the drive stack: none of the safety gates uart_backup_node
applies are here, and steering is always centered.

subscribes: cmd/raw_throttle (Int32)
"""

import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
import serial

from ros2_parts.parameters import declare


class UartBackup2Node(Node):
    def __init__(self):
        super().__init__("uart_backup2")

        port = declare(self, "port", "/dev/ttyACM0", "Serial port the VESC is on.")
        baudrate = declare(self, "baudrate", 115200, "Serial baud rate.")
        rate_hz = declare(self, "rate_hz", 10.0, "Rate commands are sent at, in Hz.")

        self.ser = serial.Serial(port=port, baudrate=baudrate)
        time.sleep(2) # sleeping to warm up vesc
        self.get_logger().info(f"Connected to {port} @ {baudrate}")
        self.get_logger().warn("bench rig: no heartbeat or RTK gating, keep the kart on stands")

        self.n = None

        self.create_subscription(Int32, "cmd/raw_throttle", self.on_command, 10)
        self.create_timer(1.0 / rate_hz, self.send_command)

    def on_command(self, msg):
        self.n = msg.data

    def run(self, n):
        self.ser.write(f"0,{n}\r".encode("ascii"))
        self.ser.flush()

    def send_command(self):
        if self.n is None:
            return
        self.run(self.n)

    def destroy_node(self):
        if self.ser is not None and self.ser.is_open:
            self.run(0)
            self.ser.close()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = UartBackup2Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
