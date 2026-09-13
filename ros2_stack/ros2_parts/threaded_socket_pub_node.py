#!/usr/bin/env python3
"""ROS 2 port of parts/threaded_socket_pub_part.py.

Streams telemetry to the host over a TCP socket without blocking. The socket
server, the single-slot queue and the newline-delimited JSON are unchanged:
the networking runs on its own thread so a host that is slow or absent never
stalls the node.

subscribes: gps/fix, gps/fused_fix (NavSatFix), odometry/filtered (Odometry),
            imu/data (Imu), cmd/steering (Float64)
"""

import json
import math
import queue
import socket
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, NavSatFix
from std_msgs.msg import Float64

from ros2_parts.orientation import quaternion_to_yaw
from ros2_parts.parameters import declare

# how long accept waits before rechecking whether to keep running
ACCEPT_TIMEOUT_S = 1.0


class ThreadedSocketPubNode(Node):
    def __init__(self):
        super().__init__("threaded_socket_pub")

        self.host = declare(self, "host", "localhost", "Interface to listen on.")
        self.port = declare(self, "port", 5005, "Port to listen on.")
        rate_hz = declare(self, "rate_hz", 10.0, "Rate telemetry is queued at, in Hz.")

        self.lat = None
        self.lon = None
        self.heading = None
        self.steer = None
        self.imu_heading = None
        self.fused_lat = None
        self.fused_lon = None

        self.data_queue = queue.Queue(maxsize=1) # Only keep the latest data point
        self.running = True

        # Start the background networking thread immediately
        self.thread = threading.Thread(target=self._network_loop, daemon=True)
        self.thread.start()

        self.create_subscription(NavSatFix, "gps/fix", self.on_fix, qos_profile_sensor_data)
        self.create_subscription(NavSatFix, "gps/fused_fix", self.on_fused_fix, 10)
        self.create_subscription(Odometry, "odometry/filtered", self.on_odometry, 10)
        self.create_subscription(Imu, "imu/data", self.on_imu, qos_profile_sensor_data)
        self.create_subscription(Float64, "cmd/steering", self.on_steering, 10)
        self.create_timer(1.0 / rate_hz, self.queue_telemetry)

    def _network_loop(self):
        """ This runs in the background and does the 'blocking' work. """
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((self.host, self.port))
        server_socket.listen(1)

        while self.running:
            self.get_logger().info(f"Waiting for host on port {self.port}...")
            # .settimeout allows us to check self.running occasionally
            server_socket.settimeout(ACCEPT_TIMEOUT_S)
            try:
                conn, addr = server_socket.accept()
            except socket.timeout:
                continue # Just loop back and check if we should still be running

            with conn:
                self.get_logger().info(f"Host connected from {addr}")
                while self.running:
                    try:
                        data = self.data_queue.get(timeout=ACCEPT_TIMEOUT_S)
                    except queue.Empty:
                        continue
                    message = json.dumps(data) + '\n'
                    try:
                        conn.sendall(message.encode('utf-8'))
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        self.get_logger().info("Host disconnected.")
                        break

        server_socket.close()

    def on_fix(self, fix):
        self.lat = fix.latitude
        self.lon = fix.longitude

    def on_fused_fix(self, fix):
        self.fused_lat = fix.latitude
        self.fused_lon = fix.longitude

    def on_odometry(self, odom):
        self.heading = math.degrees(quaternion_to_yaw(odom.pose.pose.orientation))

    def on_imu(self, imu):
        self.imu_heading = math.degrees(quaternion_to_yaw(imu.orientation))

    def on_steering(self, msg):
        self.steer = msg.data

    def run(self, lat, lon, heading, steer, imu_heading, fused_lat, fused_lon):
        """
        Non-blocking: puts data in a queue and returns immediately.
        """
        data = {"lat": lat, "lon": lon, "heading": heading, "steer": steer,
                "imu_heading": imu_heading, "fused_lat": fused_lat, "fused_lon": fused_lon}
        try:
            # If queue is full, remove old data to keep it fresh
            if self.data_queue.full():
                self.data_queue.get_nowait()
            self.data_queue.put_nowait(data)
        except (queue.Full, queue.Empty):
            pass

    def queue_telemetry(self):
        self.run(self.lat, self.lon, self.heading, self.steer,
                 self.imu_heading, self.fused_lat, self.fused_lon)

    def destroy_node(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0 * ACCEPT_TIMEOUT_S)
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ThreadedSocketPubNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
