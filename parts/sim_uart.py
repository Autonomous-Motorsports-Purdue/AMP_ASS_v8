from __future__ import annotations

from sim.kart_dynamics import KartDynamicsParams
from sim.shared_truth import SharedTruthState


class SimUARTBackupDriver:
    def __init__(self, shared_truth: SharedTruthState, params: KartDynamicsParams):
        self.shared_truth = shared_truth
        self.params = params
        self._iter = 0

    def reset_kart(self):
        self.shared_truth.set_command(throttle=0.0, steering=0.0, alive=True)

    def run(self, v, s, alive):
        if not alive:
            self.shared_truth.set_command(throttle=0.0, steering=0.0, alive=False)
            return

        throttle = 0.0 if v is None else float(v)
        steering = 0.0 if s is None else float(s)

        if self._iter < int(self.params.startup_warmup_loops):
            throttle = 0.0
            steering = 0.0

        startup_clip_loops = int(round(self.params.startup_clip_duration_s * self.params.drive_loop_hz))
        if self._iter < startup_clip_loops:
            throttle = min(throttle, float(self.params.startup_clip_throttle))

        self.shared_truth.set_command(throttle=throttle, steering=steering, alive=True)
        self._iter += 1

    def shutdown(self):
        self.shared_truth.set_command(throttle=0.0, steering=0.0, alive=False)

