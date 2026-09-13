#!/usr/bin/env python3
"""ROS 2 port of parts/control_mux.py.

The driver takes over as soon as either of their inputs is non-zero, and
their throttle is scaled back. Otherwise the autonomy stack drives.

subscribes: cmd/user_steering, cmd/user_throttle, cmd/auto_steering,
            cmd/auto_throttle (Float64)
publishes:  cmd/steering, cmd/throttle (Float64)
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64

from ros2_parts.parameters import declare


class ControlMuxNode(Node):
    def __init__(self):
        super().__init__("control_mux")

        self.user_throttle_scale = declare(
            self, "user_throttle_scale", 0.7,
            "Scale applied to the driver's throttle when they take over.")
        rate_hz = declare(self, "rate_hz", 30.0, "Rate commands are published at, in Hz.")

        self.user_steer = None
        self.user_throt = None
        self.auto_steer = None
        self.auto_throt = None
        self.was_user = False

        self.steer_pub = self.create_publisher(Float64, "cmd/steering", 10)
        self.throt_pub = self.create_publisher(Float64, "cmd/throttle", 10)

        self.create_subscription(Float64, "cmd/user_steering", self.on_user_steer, 10)
        self.create_subscription(Float64, "cmd/user_throttle", self.on_user_throt, 10)
        self.create_subscription(Float64, "cmd/auto_steering", self.on_auto_steer, 10)
        self.create_subscription(Float64, "cmd/auto_throttle", self.on_auto_throt, 10)
        self.create_timer(1.0 / rate_hz, self.publish_command)

    def on_user_steer(self, msg):
        self.user_steer = msg.data

    def on_user_throt(self, msg):
        self.user_throt = msg.data

    def on_auto_steer(self, msg):
        self.auto_steer = msg.data

    def on_auto_throt(self, msg):
        self.auto_throt = msg.data

    def run(self, user_steer, user_throt, auto_steer, auto_throt):
        if user_steer is None or user_throt is None:
            return auto_steer, auto_throt
        if user_steer != 0 or user_throt != 0:
            self.log_takeover(True)
            return user_steer, user_throt * self.user_throttle_scale
        else:
            self.log_takeover(False)
            return auto_steer, auto_throt

    def log_takeover(self, is_user):
        # only on the change, so taking over does not flood the console
        if is_user != self.was_user:
            self.get_logger().info("USER CONTROL" if is_user else "AUTO CONTROL")
            self.was_user = is_user

    def publish_command(self):
        steer, throt = self.run(
            self.user_steer, self.user_throt, self.auto_steer, self.auto_throt)
        if steer is None or throt is None:
            return
        self.steer_pub.publish(Float64(data=float(steer)))
        self.throt_pub.publish(Float64(data=float(throt)))


def main(args=None):
    rclpy.init(args=args)
    node = ControlMuxNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
