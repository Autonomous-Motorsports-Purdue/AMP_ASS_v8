#!/usr/bin/env python3
"""ROS 2 port of parts/health_check.py.

The part returned True unconditionally, with the real socket check left below
an early return. That check is implemented here and gated behind enabled,
which defaults to off so the behavior out of the box matches what was running.

publishes: safety/heartbeat (Bool)
"""

import socket

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool

from ros2_parts.parameters import declare


class HealthCheckNode(Node):
    def __init__(self):
        super().__init__("health_check")

        self.enabled = declare(
            self, "enabled", False, "Actually check the link, rather than always passing.")
        self.host_ip = declare(self, "host_ip", "192.168.12.25", "Host to check.")
        self.host_port = declare(self, "host_port", 6000, "Port the host listens on.")
        self.timeout = declare(
            self, "timeout", 0.05, "How long to wait for the host to reply, in seconds.")
        rate_hz = declare(self, "rate_hz", 30.0, "Rate the heartbeat is published at, in Hz.")

        self.clientSocket = None
        if self.enabled:
            self.connect()
        else:
            self.get_logger().warn("health check disabled, heartbeat always reports alive")

        self.pub = self.create_publisher(Bool, "safety/heartbeat", 10)
        self.create_timer(1.0 / rate_hz, self.publish_heartbeat)

    def connect(self):
        try:
            self.clientSocket = socket.create_connection(
                (self.host_ip, self.host_port), timeout=self.timeout)
            self.get_logger().info(f"connected to host {self.host_ip}:{self.host_port}")
        except OSError as e:
            self.get_logger().error(f"could not reach the host: {e}")
            self.clientSocket = None

    def run(self):
        """
        Determine if there is a still a connection between the kart and the host
        """
        if not self.enabled:
            return True
        if self.clientSocket is None:
            return False
        try:
            self.clientSocket.sendall("AMP_ALIVE\n".encode("ascii"))
            self.clientSocket.settimeout(self.timeout)
            res = self.clientSocket.recv(1024).decode()
            if res == "AMP_ALIVE\n":
                return True
            else:
                self.get_logger().warn("recieved not ok from host", throttle_duration_sec=5.0)
                return False
        except socket.timeout:
            self.get_logger().warn("socket timeout", throttle_duration_sec=5.0)
            return False
        except OSError as e:
            self.get_logger().warn(f"{e}", throttle_duration_sec=5.0)
            return False

    def publish_heartbeat(self):
        self.pub.publish(Bool(data=self.run()))

    def destroy_node(self):
        if self.clientSocket is not None:
            self.clientSocket.close()
        return super().destroy_node()


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
