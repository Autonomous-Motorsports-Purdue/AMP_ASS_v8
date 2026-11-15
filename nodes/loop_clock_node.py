#!/usr/bin/env python3
# in: none | out: loop/index, loop/monotonic_ns (Int64)
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int64


class LoopClockNode(Node):
    def __init__(self):
        super().__init__("loop_clock")
        rate_hz = self.declare_parameter("rate_hz", 50.0).value

        self.loop_index = 0
        self.index_pub = self.create_publisher(Int64, "loop/index", 10)
        self.monotonic_pub = self.create_publisher(Int64, "loop/monotonic_ns", 10)
        self.create_timer(1.0 / rate_hz, self.run)

    def run(self):
        self.loop_index += 1
        self.index_pub.publish(Int64(data=self.loop_index))
        self.monotonic_pub.publish(Int64(data=time.monotonic_ns()))


def main(args=None):
    rclpy.init(args=args)
    node = LoopClockNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
