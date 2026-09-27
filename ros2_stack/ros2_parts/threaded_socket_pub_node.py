#!/usr/bin/env python3
# in: gps/fix, gps/fused_fix (NavSatFix), odometry/filtered (Odometry), imu/data (Imu), cmd/steering (Float64) | out: newline JSON over TCP to the host
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


class ThreadedSocketPubNode(Node):
    def __init__(self):
        super().__init__("threaded_socket_pub")
        self.host = self.declare_parameter("host", "localhost").value
        self.port = self.declare_parameter("port", 5005).value
        rate_hz = self.declare_parameter("rate_hz", 10.0).value

        self.data = {"lat": None, "lon": None, "heading": None, "steer": None,
                     "imu_heading": None, "fused_lat": None, "fused_lon": None}
        self.data_queue = queue.Queue(maxsize=1)  # only keep the latest data point
        self.running = True

        # socket work blocks, so it lives on its own thread
        self.thread = threading.Thread(target=self.network_loop, daemon=True)
        self.thread.start()

        self.create_subscription(NavSatFix, "gps/fix", self.on_fix, qos_profile_sensor_data)
        self.create_subscription(NavSatFix, "gps/fused_fix", self.on_fused_fix, 10)
        self.create_subscription(Odometry, "odometry/filtered", self.on_odometry, 10)
        self.create_subscription(Imu, "imu/data", self.on_imu, qos_profile_sensor_data)
        self.create_subscription(Float64, "cmd/steering", self.on_steering, 10)
        self.create_timer(1.0 / rate_hz, self.run)

    def network_loop(self):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((self.host, self.port))
        server_socket.listen(1)
        server_socket.settimeout(1.0)  # so we can check self.running

        while self.running:
            try:
                conn, addr = server_socket.accept()
            except socket.timeout:
                continue
            with conn:
                self.get_logger().info(f"Host connected from {addr}")
                while self.running:
                    try:
                        data = self.data_queue.get(timeout=1.0)
                        conn.sendall((json.dumps(data) + "\n").encode("utf-8"))
                    except queue.Empty:
                        continue
                    except OSError:
                        self.get_logger().info("Host disconnected.")
                        break
        server_socket.close()

    def on_fix(self, fix):
        self.data["lat"] = fix.latitude
        self.data["lon"] = fix.longitude

    def on_fused_fix(self, fix):
        self.data["fused_lat"] = fix.latitude
        self.data["fused_lon"] = fix.longitude

    def on_odometry(self, odom):
        self.data["heading"] = math.degrees(quaternion_to_yaw(odom.pose.pose.orientation))

    def on_imu(self, imu):
        self.data["imu_heading"] = math.degrees(quaternion_to_yaw(imu.orientation))

    def on_steering(self, msg):
        self.data["steer"] = msg.data

    def run(self):
        # drop stale data so the host always gets the newest
        try:
            self.data_queue.get_nowait()
        except queue.Empty:
            pass
        self.data_queue.put_nowait(dict(self.data))

    def destroy_node(self):
        self.running = False
        self.thread.join(timeout=2.0)
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
