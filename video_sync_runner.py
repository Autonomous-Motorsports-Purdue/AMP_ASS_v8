import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "static_donkeycar" / "donkeycar"))

import donkeycar as dk

from parts.gstreamer_video_sync import GStreamerUvcRecorder
from parts.loop_clock import LoopClock


def main():
    parser = argparse.ArgumentParser(description="Record time-synced UVC video only.")
    parser.add_argument("--device", default="/dev/video4")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--source-format", default="mjpeg", choices=["mjpeg", "raw"])
    parser.add_argument("--output-dir", default="data/video")
    parser.add_argument("--bitrate-kbps", type=int, default=5000)
    parser.add_argument("--loop-hz", type=float, default=50.0)
    parser.add_argument("--max-loop-count", type=int, default=None)
    args = parser.parse_args()

    vehicle = dk.vehicle.Vehicle()
    vehicle.add(
        LoopClock(),
        inputs=[],
        outputs=["loop/index", "loop/monotonic_ns", "loop/wall_time"],
    )

    recorder = GStreamerUvcRecorder(
        device=args.device,
        output_dir=args.output_dir,
        width=args.width,
        height=args.height,
        fps=args.fps,
        bitrate_kbps=args.bitrate_kbps,
        source_format=args.source_format,
    )
    vehicle.add(
        recorder,
        inputs=["loop/index", "loop/monotonic_ns", "loop/wall_time"],
        outputs=[
            "video/nearest_camera_frame_id",
            "video/nearest_frame_pts_ns",
            "video/nearest_frame_monotonic_ns",
            "video/time_s",
            "video/delta_loop_to_frame_ms",
            "video/path",
        ],
        threaded=False,
    )

    vehicle.start(rate_hz=args.loop_hz, max_loop_count=args.max_loop_count)


if __name__ == "__main__":
    main()
