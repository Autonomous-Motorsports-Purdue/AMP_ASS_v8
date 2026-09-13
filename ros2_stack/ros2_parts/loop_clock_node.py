#!/usr/bin/env python3
"""ROS 2 port of parts/loop_clock.py.

In donkeycar this stamped each pass of the drive loop. Every ROS message
already carries a header stamp, so the only consumer left is
gstreamer_video_sync_node, which needs a monotonic nanosecond clock to
compare against GStreamer buffer timestamps.

publishes: loop/index, loop/monotonic_ns (Int64)
"""

import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int64

from ros2_parts.parameters import declare


class LoopClockNode(Node):
    def __init__(self):
        super().__init__("loop_clock")

        rate_hz = declare(self, "rate_hz", 50.0, "Rate ticks are published at, in Hz.")

        self.loop_index = 0
        self.index_pub = self.create_publisher(Int64, "loop/index", 10)
        self.monotonic_pub = self.create_publisher(Int64, "loop/monotonic_ns", 10)
        self.create_timer(1.0 / rate_hz, self.publish_tick)

    def run(self):
        self.loop_index += 1
        return self.loop_index, time.monotonic_ns()

    def publish_tick(self):
        loop_index, monotonic_ns = self.run()
        self.index_pub.publish(Int64(data=loop_index))
        self.monotonic_pub.publish(Int64(data=monotonic_ns))


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
