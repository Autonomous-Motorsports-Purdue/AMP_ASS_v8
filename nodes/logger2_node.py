#!/usr/bin/env python3
# in: each Float64 topic listed in the topics param | out: log.txt
from datetime import datetime
from functools import partial

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64


class Logger2Node(Node):
    def __init__(self):
        super().__init__("logger2")
        self.path = self.declare_parameter("path", "log.txt").value
        topics = self.declare_parameter("topics", ["cmd/steering", "cmd/throttle"]).value

        for topic in topics:
            self.create_subscription(Float64, topic, partial(self.run, topic), 10)

    def run(self, name, msg):
        with open(self.path, "a") as file:
            file.write(f"[{datetime.now()}] [{name}]: {msg.data}\n")


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
