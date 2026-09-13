#!/usr/bin/env python3
"""ROS 2 port of parts/test.py.

Publishes fixed control values, for bringing the drive chain up by itself.
The module is named part_test_node rather than test_node so pytest does not
collect it.

publishes: cmd/user_steering, cmd/user_throttle (Float64)
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64

from ros2_parts.parameters import declare


class PartTestNode(Node):
    def __init__(self):
        super().__init__("test")

        self.steering = declare(self, "steering", 0.0, "Constant steering published.")
        self.throttle = declare(self, "throttle", 0.9921875, "Constant throttle published.")
        rate_hz = declare(self, "rate_hz", 30.0, "Rate commands are published at, in Hz.")

        self.steer_pub = self.create_publisher(Float64, "cmd/user_steering", 10)
        self.throt_pub = self.create_publisher(Float64, "cmd/user_throttle", 10)
        self.create_timer(1.0 / rate_hz, self.publish_command)

    def run(self):
        return self.steering, self.throttle

    def publish_command(self):
        steering, throttle = self.run()
        self.steer_pub.publish(Float64(data=float(steering)))
        self.throt_pub.publish(Float64(data=float(throttle)))


def main(args=None):
    rclpy.init(args=args)
    node = PartTestNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
