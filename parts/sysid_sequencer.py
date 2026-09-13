import math


class SysIDSequencer:
    """Plays back open-loop segments against the loop clock.

    segment: {t: seconds, thr: counts, steer: -1..1}
          or {t: seconds, thr: counts, chirp: {amp, f0, f1}}  (linear sweep)
    Missing thr/steer default to 0. Steer output = steer_bias + segment steer.
    """

    def __init__(self, segments, steer_bias=0.0, max_speed_mps=6.0):
        self.segments = segments
        self.steer_bias = steer_bias
        self.max_speed_mps = max_speed_mps
        self.t0 = None
        self.overspeed = False

    @property
    def duration_s(self):
        return sum(s["t"] for s in self.segments)

    def run(self, monotonic_ns, gps_speed):
        if self.t0 is None:
            self.t0 = monotonic_ns
        self.overspeed |= (gps_speed or 0.0) > self.max_speed_mps

        idx, t = 0, (monotonic_ns - self.t0) / 1e9
        while idx < len(self.segments) - 1 and t >= self.segments[idx]["t"]:
            t -= self.segments[idx]["t"]
            idx += 1
        seg = self.segments[idx]

        steer = seg.get("steer", 0.0)
        if "chirp" in seg:
            c = seg["chirp"]
            k = (c["f1"] - c["f0"]) / seg["t"]
            steer = c["amp"] * math.sin(2 * math.pi * (c["f0"] * t + 0.5 * k * t * t))
        throttle = 0 if self.overspeed else seg.get("thr", 0)
        return throttle, self.steer_bias + steer, idx, t
