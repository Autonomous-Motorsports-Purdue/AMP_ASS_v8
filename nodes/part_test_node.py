#!/usr/bin/env python3
# in: none | out: cmd/user_steering, cmd/user_throttle (Float64) -- constant values
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64


class PartTestNode(Node):
    def __init__(self):
        super().__init__("test")
        rate_hz = self.declare_parameter("rate_hz", 30.0).value
        self.steer_pub = self.create_publisher(Float64, "cmd/user_steering", 10)
        self.throt_pub = self.create_publisher(Float64, "cmd/user_throttle", 10)
        self.create_timer(1.0 / rate_hz, self.run)

    def run(self):
        self.steer_pub.publish(Float64(data=0.0))
        self.throt_pub.publish(Float64(data=0.9921875))


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
