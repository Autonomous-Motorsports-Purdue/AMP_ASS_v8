#!/usr/bin/env python3
# in: cmd/raw_throttle (Int32) | out: raw "0,n" serial to the VESC (bench test only, no safety gating)
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
import serial


class UartBackup2Node(Node):
    def __init__(self):
        super().__init__("uart_backup2")
        port = self.declare_parameter("port", "/dev/ttyACM0").value
        rate_hz = self.declare_parameter("rate_hz", 10.0).value

        self.ser = serial.Serial(port=port, baudrate=115200)
        time.sleep(2)  # let the VESC warm up
        self.n = None

        self.create_subscription(Int32, "cmd/raw_throttle", self.on_command, 10)
        self.create_timer(1.0 / rate_hz, self.run)

    def on_command(self, msg):
        self.n = msg.data

    def run(self):
        if self.n is None:
            return
        self.ser.write(f"0,{self.n}\r".encode("ascii"))
        self.ser.flush()

    def destroy_node(self):
        self.ser.write(b"0,0\r")
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
