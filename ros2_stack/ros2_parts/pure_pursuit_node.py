#!/usr/bin/env python3
"""ROS 2 port of parts/pure_pursuit.py.

Steers toward a single ground-plane target point.

subscribes: perception/waypoint_ground (PointStamped)
publishes:  cmd/auto_steering (Float64, normalized -1 to 1),
            cmd/auto_throttle (Float64)
"""

import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float64

from ros2_parts.parameters import declare


class PurePursuitNode(Node):
    def __init__(self):
        super().__init__("pure_pursuit")

        self.wheelbase = declare(self, "wheelbase", 1.000506, "Vehicle wheelbase, in meters.")
        self.speed = declare(self, "speed", 0.3, "Constant throttle returned.")
        self.speed_fast = declare(self, "speed_fast", 0.45, "Throttle on a straightaway.")
        self.speed_prev = self.speed_fast

        self.steer_pub = self.create_publisher(Float64, "cmd/auto_steering", 10)
        self.throt_pub = self.create_publisher(Float64, "cmd/auto_throttle", 10)
        self.create_subscription(
            PointStamped, "perception/waypoint_ground", self.on_waypoint, 10)

    def run(self, target_position):
        targetx, targety, _ = target_position
        targety = -targety
        targetx += 0.6096 #Back wheels to camera
        targetx *= 0.75 # Constant for urgency
        # we move forward in x
        yaw = 0

        alpha = math.atan2(targety, targetx) - (yaw * (math.pi/180.0)) # Angle from rear of car to target. good.

        Ld = math.sqrt(targetx**2 + targety**2)
        if Ld < 1e-6:
            return None, None

        steering_theta  = math.atan2((2*self.wheelbase*math.sin(alpha)),Ld) # Steering angle in radians.
        steering_theta *= (180.0/math.pi)
        steering_value = steering_theta / 18.523

        # speed control!
        if abs(steering_theta) < 3.0: # if absolute value of steering less than 1.0
            ret_speed = self.speed_fast * 0.5 + 0.5 * self.speed_prev
        else:
            ret_speed = self.speed

        self.speed_prev = ret_speed
        ret_speed = self.speed

        return steering_value, ret_speed

    def on_waypoint(self, msg):
        steering_value, ret_speed = self.run((msg.point.x, msg.point.y, msg.point.z))
        if steering_value is None:
            self.get_logger().warn("target is on the rear axle", throttle_duration_sec=5.0)
            return

        self.steer_pub.publish(Float64(data=float(steering_value)))
        self.throt_pub.publish(Float64(data=float(ret_speed)))


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
