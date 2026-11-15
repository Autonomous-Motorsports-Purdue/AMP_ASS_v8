#!/usr/bin/env python3
# in: Velodyne UDP packets | out: lidar/points (PointCloud2), lidar/occupancy_grid (OccupancyGrid)
import math
import socket
import time

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2, PointField
import velodyne_decoder as vd


class LidarNode(Node):
    def __init__(self):
        super().__init__("lidar")
        port = self.declare_parameter("port", 2368).value
        bind_address = self.declare_parameter("bind_address", "0.0.0.0").value
        receive_buffer_bytes = int(
            self.declare_parameter("receive_buffer_bytes", 4 * 1024 * 1024).value
        )
        self.resolution = float(self.declare_parameter("resolution", 0.1).value)
        self.grid_range = float(self.declare_parameter("grid_range", 29.0).value)
        self.height_min = float(self.declare_parameter("height_min", -0.25).value)
        self.height_max = float(self.declare_parameter("height_max", 2.0).value)
        rate_hz = float(self.declare_parameter("rate_hz", 10.0).value)
        self.frame_id = self.declare_parameter("frame_id", "velodyne").value

        if not 1 <= int(port) <= 65535:
            raise ValueError(f"port must be between 1 and 65535; got {port}")
        if not isinstance(bind_address, str) or not bind_address:
            raise ValueError("bind_address must be a non-empty IPv4 address")
        if receive_buffer_bytes <= 0:
            raise ValueError("receive_buffer_bytes must be greater than zero")
        if not math.isfinite(self.resolution) or self.resolution <= 0.0:
            raise ValueError("resolution must be greater than zero")
        if not math.isfinite(self.grid_range) or self.grid_range <= 0.0:
            raise ValueError("grid_range must be greater than zero")
        if (not math.isfinite(self.height_min) or not math.isfinite(self.height_max)
                or self.height_min >= self.height_max):
            raise ValueError("height_min must be less than height_max")
        if not math.isfinite(rate_hz) or rate_hz <= 0.0:
            raise ValueError("rate_hz must be greater than zero")

        self.grid_size = int((2.0 * self.grid_range) / self.resolution)
        if self.grid_size < 1:
            raise ValueError("grid_range and resolution produce an empty grid")

        self.config = vd.Config(min_range=0, max_range=self.grid_range)
        self.decoder = vd.StreamDecoder(self.config)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.setsockopt(
            socket.SOL_SOCKET, socket.SO_RCVBUF, receive_buffer_bytes
        )
        self.sock.bind((bind_address, int(port)))
        self.sock.setblocking(False)

        self.packets_received = 0
        self.scans_decoded = 0
        self.last_sender = None
        self.started_at = time.monotonic()

        self.points_pub = self.create_publisher(
            PointCloud2, "lidar/points", qos_profile_sensor_data)
        self.grid_pub = self.create_publisher(
            OccupancyGrid, "lidar/occupancy_grid", qos_profile_sensor_data)
        self.create_timer(1.0 / rate_hz, self.run)
        self.create_timer(5.0, self._report_status)
        actual_buffer = self.sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
        self.get_logger().info(
            f"Listening for Velodyne data on UDP {bind_address}:{port} "
            f"(receive buffer: {actual_buffer} bytes)"
        )

    def _report_status(self):
        if self.packets_received == 0:
            elapsed = int(time.monotonic() - self.started_at)
            self.get_logger().warning(
                f"No Velodyne UDP packets received after {elapsed}s. Check that "
                "the sensor destination IP is the host Ethernet address, UDP "
                "2368 is allowed by the host firewall, and the container uses "
                "host networking."
            )
        elif self.scans_decoded == 0:
            self.get_logger().warning(
                f"Received {self.packets_received} UDP packets from "
                f"{self.last_sender}, but no complete Velodyne scan decoded. "
                "Check the LiDAR model/data format."
            )

    def _filtered_points(self, points):
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] < 3:
            self.get_logger().warning(
                "Velodyne decoder returned points without x, y, z columns",
                throttle_duration_sec=5.0,
            )
            return np.empty((0, 4), dtype=np.float32)

        points = points[np.isfinite(points[:, :3]).all(axis=1)]

        result = np.zeros((len(points), 4), dtype=np.float32)
        result[:, :3] = points[:, :3]
        if points.shape[1] > 3:
            intensity = points[:, 3]
            result[:, 3] = np.where(np.isfinite(intensity), intensity, 0.0)
        return result

    def _publish_points(self, points, stamp):
        msg = PointCloud2()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id
        msg.height = 1
        msg.width = len(points)
        msg.fields = [
            PointField(name=name, offset=offset, datatype=PointField.FLOAT32, count=1)
            for name, offset in (("x", 0), ("y", 4), ("z", 8), ("intensity", 12))
        ]
        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = True
        msg.data = points.tobytes()
        self.points_pub.publish(msg)

    def _publish_grid(self, points, stamp):
        grid = np.full((self.grid_size, self.grid_size), -1, dtype=np.int8)
        in_bounds = (points[:, 2] > self.height_min) & (points[:, 2] < self.height_max)
        in_bounds &= (np.abs(points[:, 0]) < self.grid_range)
        in_bounds &= (np.abs(points[:, 1]) < self.grid_range)
        points_in_bounds = points[in_bounds]
        if len(points_in_bounds):
            grid_x = (
                (points_in_bounds[:, 0] + self.grid_range) / self.resolution
            ).astype(np.int32)
            grid_y = (
                (points_in_bounds[:, 1] + self.grid_range) / self.resolution
            ).astype(np.int32)
            grid_x = np.clip(grid_x, 0, self.grid_size - 1)
            grid_y = np.clip(grid_y, 0, self.grid_size - 1)
            grid[grid_y, grid_x] = 100

        msg = OccupancyGrid()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id
        msg.info.resolution = self.resolution
        msg.info.width = self.grid_size
        msg.info.height = self.grid_size
        msg.info.origin.position.x = -self.grid_range
        msg.info.origin.position.y = -self.grid_range
        msg.info.origin.orientation.w = 1.0
        msg.data = grid.ravel().tolist()
        self.grid_pub.publish(msg)

    def run(self):
        latest_points = None
        while True:
            try:
                data, address = self.sock.recvfrom(65535)
            except BlockingIOError:
                break
            except OSError as exc:
                self.get_logger().error(
                    f"Failed to receive Velodyne UDP data: {exc}",
                    throttle_duration_sec=5.0,
                )
                break

            self.packets_received += 1
            self.last_sender = f"{address[0]}:{address[1]}"
            try:
                scan = self.decoder.decode(time.time(), data)
            except Exception as exc:
                self.get_logger().warning(
                    f"Velodyne packet could not be decoded: {exc}",
                    throttle_duration_sec=5.0,
                )
                continue
            if scan is not None and len(scan[1]) > 0:
                latest_points = scan[1]
                self.scans_decoded += 1

        if latest_points is None:
            return

        points = self._filtered_points(latest_points)
        stamp = self.get_clock().now().to_msg()
        self._publish_points(points, stamp)
        self._publish_grid(points, stamp)

    def destroy_node(self):
        self.sock.close()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LidarNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
