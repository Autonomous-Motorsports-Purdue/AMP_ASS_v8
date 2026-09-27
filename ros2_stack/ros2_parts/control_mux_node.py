#!/usr/bin/env python3
# in: cmd/user_steering, cmd/user_throttle, cmd/auto_steering, cmd/auto_throttle (Float64) | out: cmd/steering, cmd/throttle (Float64)
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64


class ControlMuxNode(Node):
    def __init__(self):
        super().__init__("control_mux")
        self.user_throttle_scale = self.declare_parameter("user_throttle_scale", 0.7).value
        rate_hz = self.declare_parameter("rate_hz", 30.0).value

        self.user_steer = None
        self.user_throt = None
        self.auto_steer = None
        self.auto_throt = None

        self.create_subscription(Float64, "cmd/user_steering", self.on_user_steer, 10)
        self.create_subscription(Float64, "cmd/user_throttle", self.on_user_throt, 10)
        self.create_subscription(Float64, "cmd/auto_steering", self.on_auto_steer, 10)
        self.create_subscription(Float64, "cmd/auto_throttle", self.on_auto_throt, 10)
        self.steer_pub = self.create_publisher(Float64, "cmd/steering", 10)
        self.throt_pub = self.create_publisher(Float64, "cmd/throttle", 10)
        self.create_timer(1.0 / rate_hz, self.run)

    def on_user_steer(self, msg):
        self.user_steer = msg.data

    def on_user_throt(self, msg):
        self.user_throt = msg.data

    def on_auto_steer(self, msg):
        self.auto_steer = msg.data

    def on_auto_throt(self, msg):
        self.auto_throt = msg.data

    def run(self):
        steer, throt = self.auto_steer, self.auto_throt
        if self.user_steer is not None and self.user_throt is not None:
            if self.user_steer != 0 or self.user_throt != 0:
                self.get_logger().info(
                    f"USER CONTROL: {self.user_steer}, {self.user_throt}", throttle_duration_sec=1.0)
                steer, throt = self.user_steer, self.user_throt * self.user_throttle_scale

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
