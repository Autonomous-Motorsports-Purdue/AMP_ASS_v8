import math

class KinematicBicycleModel:

    def __init__(self, x, y, heading, wheelbase=1, dt=0.02):
        self.x = float(x)
        self.y = float(y)
        self.heading = float(heading)
        self.wheelbase = float(wheelbase)
        self.dt = float(dt)

    def update(self, speed, steer, max_steer_deg=30.0):
        steer_angle = steer * math.radians(max_steer_deg)
        self.x += speed * math.cos(self.heading) * self.dt
        self.y += speed * math.sin(self.heading) * self.dt
        self.heading += (speed / self.wheelbase) * math.tan(steer_angle) * self.dt
        self.heading = (self.heading + math.pi) % (2 * math.pi) - math.pi