#!/usr/bin/env python3
# in: cmd/throttle, cmd/steering (Float64, -1 to 1), safety/heartbeat (Bool) | out: framed binary serial to the kart
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float64
import serial

START_BYTE = 254
END_BYTE = 255


class UartNode(Node):
    def __init__(self):
        super().__init__("uart")
        port_name = self.declare_parameter("port_name", "/dev/ttyACM0").value
        rate_hz = self.declare_parameter("rate_hz", 30.0).value

        self.ser = serial.Serial(port=port_name, baudrate=115200)
        self.curr_v = 0
        self.curr_s = 0

        self.throttle = 0.0
        self.steering = 0.0
        self.alive = False

        self.create_subscription(Float64, "cmd/throttle", self.on_throttle, 10)
        self.create_subscription(Float64, "cmd/steering", self.on_steering, 10)
        self.create_subscription(Bool, "safety/heartbeat", self.on_heartbeat, 10)
        self.create_timer(1.0 / rate_hz, self.run)

    def on_throttle(self, msg):
        self.throttle = msg.data

    def on_steering(self, msg):
        self.steering = msg.data

    def on_heartbeat(self, msg):
        self.alive = msg.data

    def update_velocity(self, new_v):
        new_v = max(0, min(255, new_v))
        self.curr_v = new_v >> 1

    def update_steering(self, new_s):
        if new_s <= -63:
            self.curr_s = 0
        elif new_s >= 64:
            self.curr_s = 127
        else:
            self.curr_s = new_s + 64

    def reset_kart(self):
        self.update_velocity(0)
        self.update_steering(0)
        self.write_serial()

    def write_serial(self):
        self.ser.write(bytes([START_BYTE, self.curr_v, self.curr_s, END_BYTE]))

    def run(self):
        # no heartbeat yet counts as dead
        if not self.alive:
            self.reset_kart()
            return
        v = max(-100, min(100, int(self.throttle * 127)))
        s = int(self.steering * 127)
        self.update_velocity(v)
        self.update_steering(s)
        self.write_serial()

    def destroy_node(self):
        self.reset_kart()
        self.ser.close()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = UartNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
