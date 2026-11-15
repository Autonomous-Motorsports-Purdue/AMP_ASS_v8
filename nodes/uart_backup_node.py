#!/usr/bin/env python3
# in: cmd/throttle (Float64, eRPM), cmd/steering (Float64, -1 to 1), safety/heartbeat (Bool), gps/fix_type (String) | out: ASCII serial to kart, cmd/commanded_steering, cmd/commanded_throttle (Float64)
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float64, String
import serial


class UartBackupNode(Node):
    def __init__(self):
        super().__init__("uart_backup")
        port_name = self.declare_parameter("port_name", "/dev/ttyACM1").value
        rate_hz = self.declare_parameter("rate_hz", 50.0).value
        self.startup_cycles = self.declare_parameter("startup_cycles", 250).value
        self.startup_throttle = self.declare_parameter("startup_throttle", 1500.0).value
        self.max_steer_diff = self.declare_parameter("max_steer_diff", 0.1).value

        self.ser = serial.Serial(port=port_name, baudrate=115200)
        self._iter = 0
        self.prev_steer = 0.0

        self.throttle = 0.0
        self.steering = 0.0
        self.alive = False
        self.fix = None

        self.create_subscription(Float64, "cmd/throttle", self.on_throttle, 10)
        self.create_subscription(Float64, "cmd/steering", self.on_steering, 10)
        self.create_subscription(Bool, "safety/heartbeat", self.on_heartbeat, 10)
        self.create_subscription(String, "gps/fix_type", self.on_fix_type, 10)
        self.commanded_steer_pub = self.create_publisher(Float64, "cmd/commanded_steering", 10)
        self.commanded_throt_pub = self.create_publisher(Float64, "cmd/commanded_throttle", 10)
        self.create_timer(1.0 / rate_hz, self.run)

    def on_throttle(self, msg):
        self.throttle = msg.data

    def on_steering(self, msg):
        self.steering = msg.data

    def on_heartbeat(self, msg):
        self.alive = msg.data

    def on_fix_type(self, msg):
        self.fix = msg.data

    def write_serial(self, v, s):
        # steering -1..1 maps to 0..255
        self.ser.write(f"{int(v)},{int(127.5 * (s + 1))}\r".encode("ascii"))
        self.ser.flush()

    def run(self):
        if not self.alive:
            self.write_serial(0, 0)
            return
        self._iter += 1
        v, s = self.throttle, self.steering

        # only drive with an RTK fix
        if self.fix is None or self.fix.strip() not in ("RTK FLOAT", "RTK FIXED"):
            v = 0
            self.get_logger().warn("WAITING FOR RTK FIX", throttle_duration_sec=2.0)

        # go straight and slow for the first few seconds
        if self._iter < self.startup_cycles:
            v = min(v, self.startup_throttle)
            s = 0.0

        # slew rate limit on steering
        s = float(np.clip(s, self.prev_steer - self.max_steer_diff, self.prev_steer + self.max_steer_diff))
        self.prev_steer = s

        self.write_serial(v, s)
        self.commanded_steer_pub.publish(Float64(data=s))
        self.commanded_throt_pub.publish(Float64(data=float(v)))

    def destroy_node(self):
        self.write_serial(0, 0)
        self.ser.close()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = UartBackupNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
