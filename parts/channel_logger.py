import csv
import datetime
import os


class ChannelLogger:
    """Writes one CSV row per loop from the channels it is given.

    fields: list of channel names; a (name, width) tuple declares a
    tuple-valued channel that is flattened to name_0 .. name_{width-1}.
    Pass `logger.inputs` to V.add so header and data cannot drift.
    """

    def __init__(self, fields, path):
        self.fields = [f if isinstance(f, tuple) else (f, 1) for f in fields]
        header = [f"{n}_{i}" if w > 1 else n for n, w in self.fields for i in range(w)]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.file = open(path, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(["wall_time"] + header)

    @property
    def inputs(self):
        return [n for n, _ in self.fields]

    def run(self, *values):
        row = [datetime.datetime.now().isoformat(timespec="microseconds")]
        for (_, width), v in zip(self.fields, values):
            row += [v] if width == 1 else list(v if v is not None else [None] * width)
        self.writer.writerow(row)
        self.file.flush()

    def shutdown(self):
        self.file.close()
