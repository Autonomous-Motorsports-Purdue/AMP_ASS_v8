#!/usr/bin/env python3
# in: TCP reply from host (only when enabled) | out: safety/heartbeat (Bool)
import socket

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool


class HealthCheckNode(Node):
    def __init__(self):
        super().__init__("health_check")
        self.enabled = self.declare_parameter("enabled", False).value
        host_ip = self.declare_parameter("host_ip", "192.168.12.25").value
        host_port = self.declare_parameter("host_port", 6000).value
        rate_hz = self.declare_parameter("rate_hz", 30.0).value

        # the donkeycar part always returned True; the real check only runs when enabled
        self.clientSocket = None
        if self.enabled:
            self.clientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.clientSocket.connect((host_ip, host_port))

        self.pub = self.create_publisher(Bool, "safety/heartbeat", 10)
        self.create_timer(1.0 / rate_hz, self.run)

    def run(self):
        self.pub.publish(Bool(data=self.is_host_alive()))

    def is_host_alive(self):
        if not self.enabled:
            return True
        try:
            self.clientSocket.sendall("AMP_ALIVE\n".encode("ascii"))
            self.clientSocket.settimeout(0.05)
            res = self.clientSocket.recv(1024).decode()
            return res == "AMP_ALIVE\n"
        except Exception as e:
            self.get_logger().warn(str(e), throttle_duration_sec=5.0)
            return False


def main(args=None):
    rclpy.init(args=args)
    node = HealthCheckNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
