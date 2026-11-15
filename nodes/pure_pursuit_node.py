#!/usr/bin/env python3
# in: perception/waypoint_ground (PointStamped) | out: cmd/auto_steering (Float64, -1 to 1), cmd/auto_throttle (Float64)
import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float64


class PurePursuitNode(Node):
    def __init__(self):
        super().__init__("pure_pursuit")
        self.wheelbase = self.declare_parameter("wheelbase", 1.000506).value
        self.speed = self.declare_parameter("speed", 0.3).value

        self.create_subscription(PointStamped, "perception/waypoint_ground", self.run, 10)
        self.steer_pub = self.create_publisher(Float64, "cmd/auto_steering", 10)
        self.throt_pub = self.create_publisher(Float64, "cmd/auto_throttle", 10)

    def run(self, msg):
        targety = -msg.point.y
        targetx = msg.point.x + 0.6096  # back wheels to camera
        targetx *= 0.75  # constant for urgency

        alpha = math.atan2(targety, targetx)  # angle from rear axle to target
        Ld = math.hypot(targetx, targety)
        if Ld < 1e-6:
            return

        steering_theta = math.degrees(math.atan2(2 * self.wheelbase * math.sin(alpha), Ld))
        steering_value = steering_theta / 18.523  # degrees -> normalized steering

        self.steer_pub.publish(Float64(data=steering_value))
        self.throt_pub.publish(Float64(data=float(self.speed)))


def main(args=None):
    rclpy.init(args=args)
    node = PurePursuitNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
