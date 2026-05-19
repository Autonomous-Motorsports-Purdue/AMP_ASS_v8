import datetime
import time


class LoopClock:
    def __init__(self):
        self.loop_index = 0

    def run(self):
        self.loop_index += 1
        return (
            self.loop_index,
            time.monotonic_ns(),
            datetime.datetime.now().isoformat(timespec="microseconds"),
        )
