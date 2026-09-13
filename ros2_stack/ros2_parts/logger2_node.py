#!/usr/bin/env python3
"""ROS 2 port of parts/logger2.py.

The part took its values as kwargs, which the vehicle loop never supplied.
Here the topics to follow are named by parameter. For anything worth keeping,
ros2 bag record is the better tool.

subscribes: every topic named in the topics parameter (Float64)
"""

from datetime import datetime
from functools import partial

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64

from ros2_parts.parameters import declare


class Logger2Node(Node):
    def __init__(self):
        super().__init__("logger2")

        self.path = declare(self, "path", "log.txt", "Text file the values go to.")
        topics = declare(
            self, "topics", ["cmd/steering", "cmd/throttle"],
            "Float64 topics to log, one line per message.")

        for topic in topics:
            self.create_subscription(Float64, topic, partial(self.on_value, topic), 10)
        self.get_logger().info(f"Logging {list(topics)} to {self.path}")

        # held open rather than reopened per message, which the part did
        self.file = open(self.path, "a")

    def run(self, **kwargs):
        for name, arg in kwargs.items():
            self.file.write(f"[{datetime.now()}] [{name}]: {arg}\n")
        self.file.flush()

    def on_value(self, topic, msg):
        self.run(**{topic: msg.data})

    def destroy_node(self):
        if self.file is not None and not self.file.closed:
            self.file.close()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = Logger2Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
