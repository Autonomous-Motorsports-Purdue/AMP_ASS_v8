#!/usr/bin/env python3
"""ROS 2 port of parts/gstreamer_video_sync.py.

Records the UVC camera to MP4 and matches recorded frames to loop ticks. The
pipeline, the PTS-to-monotonic offset and the sync CSV are unchanged.

The camera is encoded straight to disk by GStreamer rather than being
published as images, so the recording keeps full frame rate regardless of what
the rest of the graph is doing. What is published is which recorded frame
lines up with each loop tick, so the video can be replayed against the log.

subscribes: loop/index, loop/monotonic_ns (Int64)
publishes:  video/frame_id (Int64), video/time_s, video/delta_ms (Float64),
            video/path (String)
"""

import csv
import datetime
import os
import threading
import time
from collections import deque
from dataclasses import dataclass

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Int64, String

from ros2_parts.parameters import declare

SYSTEM_GSTREAMER_PLUGIN_PATH = "/usr/lib/x86_64-linux-gnu/gstreamer-1.0"


@dataclass(frozen=True)
class CameraFrameStamp:
    frame_id: int
    pts_ns: int
    monotonic_ns: int

    @property
    def video_time_s(self):
        return self.pts_ns / 1_000_000_000.0


class FrameSyncIndex:
    def __init__(self, max_frames=300):
        self.frames = deque(maxlen=max_frames)

    def add_frame(self, frame_id, pts_ns, monotonic_ns):
        stamp = CameraFrameStamp(
            frame_id=int(frame_id),
            pts_ns=int(pts_ns),
            monotonic_ns=int(monotonic_ns),
        )
        self.frames.append(stamp)
        return stamp

    def nearest(self, loop_monotonic_ns):
        if loop_monotonic_ns is None or not self.frames:
            return None
        loop_monotonic_ns = int(loop_monotonic_ns)
        return min(
            self.frames,
            key=lambda frame: abs(frame.monotonic_ns - loop_monotonic_ns),
        )


class SyncCsvWriter:
    fields = [
        "loop_index",
        "loop_monotonic_ns",
        "loop_wall_time",
        "nearest_camera_frame_id",
        "nearest_frame_pts_ns",
        "nearest_frame_monotonic_ns",
        "video_time_s",
        "delta_loop_to_frame_ms",
        "video_path",
    ]

    def __init__(self, csv_path):
        self.csv_path = csv_path
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        self.csvfile = open(csv_path, "w", newline="")
        self.writer = csv.DictWriter(self.csvfile, fieldnames=self.fields)
        self.writer.writeheader()

    def write_row(self, loop_index, loop_monotonic_ns, loop_wall_time, frame, video_path):
        row = {
            "loop_index": loop_index,
            "loop_monotonic_ns": loop_monotonic_ns,
            "loop_wall_time": loop_wall_time,
            "nearest_camera_frame_id": "",
            "nearest_frame_pts_ns": "",
            "nearest_frame_monotonic_ns": "",
            "video_time_s": "",
            "delta_loop_to_frame_ms": "",
            "video_path": video_path,
        }

        if frame is not None:
            row.update(
                {
                    "nearest_camera_frame_id": frame.frame_id,
                    "nearest_frame_pts_ns": frame.pts_ns,
                    "nearest_frame_monotonic_ns": frame.monotonic_ns,
                    "video_time_s": frame.video_time_s,
                    "delta_loop_to_frame_ms": (
                        (int(loop_monotonic_ns) - frame.monotonic_ns) / 1_000_000.0
                        if loop_monotonic_ns is not None
                        else ""
                    ),
                }
            )

        self.writer.writerow(row)
        self.csvfile.flush()
        return row

    def close(self):
        if not self.csvfile.closed:
            self.csvfile.flush()
            self.csvfile.close()


class GStreamerUvcRecorder:
    def __init__(
        self,
        device="/dev/video4",
        output_dir="data/video",
        width=1280,
        height=720,
        fps=30,
        bitrate_kbps=5000,
        max_index_frames=300,
        source_format="mjpeg",
    ):
        self.device = device
        self.output_dir = output_dir
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.bitrate_kbps = int(bitrate_kbps)
        self.source_format = source_format.lower()
        self.sync_index = FrameSyncIndex(max_frames=max_index_frames)
        self.lock = threading.Lock()
        self.frame_id = 0
        self.pts_to_monotonic_offset_ns = None
        self.pipeline = None
        self.appsink = None
        self.csv_writer = None
        self.Gst = self._import_gst()

        os.makedirs(output_dir, exist_ok=True)
        start_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.video_path = os.path.join(output_dir, f"{start_time}.mp4")
        self.sync_csv_path = os.path.join(output_dir, f"{start_time}_sync.csv")

        self._start_pipeline()
        self.csv_writer = SyncCsvWriter(self.sync_csv_path)

    def _import_gst(self):
        if os.path.isdir(SYSTEM_GSTREAMER_PLUGIN_PATH):
            os.environ.setdefault("GST_PLUGIN_SYSTEM_PATH", SYSTEM_GSTREAMER_PLUGIN_PATH)

        import gi

        gi.require_version("Gst", "1.0")
        gi.require_version("GstApp", "1.0")
        from gi.repository import Gst

        Gst.init(None)
        if os.path.isdir(SYSTEM_GSTREAMER_PLUGIN_PATH):
            Gst.Registry.get().scan_path(SYSTEM_GSTREAMER_PLUGIN_PATH)
        return Gst

    def _gst_quote(self, value):
        return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'

    def _build_pipeline(self):
        video_path = self._gst_quote(os.path.abspath(self.video_path))
        device = self._gst_quote(self.device)
        source = self._build_source_pipeline(device)
        return (
            f"{source} "
            "tee name=t "
            "t. ! queue ! videoconvert ! "
            f"x264enc tune=zerolatency speed-preset=ultrafast bitrate={self.bitrate_kbps} "
            f"key-int-max={self.fps} ! "
            "h264parse ! mp4mux faststart=true ! "
            f"filesink location={video_path} "
            "t. ! queue leaky=downstream max-size-buffers=2 ! "
            "videoconvert ! video/x-raw,format=RGB ! "
            "appsink name=sync_sink emit-signals=true sync=false max-buffers=2 drop=true"
        )

    def _build_source_pipeline(self, device):
        if self.source_format in ("mjpeg", "jpeg", "image/jpeg"):
            return (
                f"v4l2src device={device} do-timestamp=true ! "
                f"image/jpeg,width={self.width},height={self.height},framerate={self.fps}/1 ! "
                "jpegdec ! videoconvert !"
            )

        if self.source_format in ("raw", "yuy2", "video/x-raw"):
            return (
                f"v4l2src device={device} do-timestamp=true ! "
                f"video/x-raw,width={self.width},height={self.height},framerate={self.fps}/1 ! "
                "videoconvert !"
            )

        raise ValueError(
            f"Unsupported source_format={self.source_format!r}; use 'mjpeg' or 'raw'."
        )

    def _start_pipeline(self):
        try:
            self.pipeline = self.Gst.parse_launch(self._build_pipeline())
        except Exception as exc:
            raise RuntimeError(
                "Unable to create the GStreamer UVC recording pipeline. "
                "Check that the camera, x264 encoder, mp4 muxer, and GStreamer plugins are installed."
            ) from exc

        self.appsink = self.pipeline.get_by_name("sync_sink")
        if self.appsink is None:
            raise RuntimeError("GStreamer pipeline did not expose the sync_sink appsink.")

        self.appsink.connect("new-sample", self._on_new_sample)
        result = self.pipeline.set_state(self.Gst.State.PLAYING)
        if result == self.Gst.StateChangeReturn.FAILURE:
            raise RuntimeError(
                f"Unable to start GStreamer recording from {self.device}. "
                "Check the camera device path and permissions."
            )

    def _on_new_sample(self, sink):
        sample = sink.emit("pull-sample")
        if sample is None:
            return self.Gst.FlowReturn.OK

        buffer = sample.get_buffer()
        pts_ns = buffer.pts
        if pts_ns == self.Gst.CLOCK_TIME_NONE:
            pts_ns = buffer.dts
        if pts_ns == self.Gst.CLOCK_TIME_NONE:
            return self.Gst.FlowReturn.OK

        now_ns = time.monotonic_ns()
        with self.lock:
            if self.pts_to_monotonic_offset_ns is None:
                self.pts_to_monotonic_offset_ns = now_ns - int(pts_ns)
            monotonic_ns = int(pts_ns) + self.pts_to_monotonic_offset_ns
            self.frame_id += 1
            self.sync_index.add_frame(self.frame_id, int(pts_ns), monotonic_ns)

        return self.Gst.FlowReturn.OK

    def _raise_pipeline_errors(self):
        if self.pipeline is None:
            return
        bus = self.pipeline.get_bus()
        while True:
            message = bus.pop()
            if message is None:
                return
            if message.type == self.Gst.MessageType.ERROR:
                err, debug = message.parse_error()
                raise RuntimeError(f"GStreamer recording error: {err}; debug={debug}")

    def run(self, loop_index, loop_monotonic_ns, loop_wall_time):
        self._raise_pipeline_errors()
        with self.lock:
            frame = self.sync_index.nearest(loop_monotonic_ns)

        row = self.csv_writer.write_row(
            loop_index=loop_index,
            loop_monotonic_ns=loop_monotonic_ns,
            loop_wall_time=loop_wall_time,
            frame=frame,
            video_path=self.video_path,
        )

        return (
            row["nearest_camera_frame_id"],
            row["nearest_frame_pts_ns"],
            row["nearest_frame_monotonic_ns"],
            row["video_time_s"],
            row["delta_loop_to_frame_ms"],
            row["video_path"],
        )

    def shutdown(self):
        if self.pipeline is not None:
            self.pipeline.send_event(self.Gst.Event.new_eos())
            bus = self.pipeline.get_bus()
            bus.timed_pop_filtered(
                2 * self.Gst.SECOND,
                self.Gst.MessageType.EOS | self.Gst.MessageType.ERROR,
            )
            self.pipeline.set_state(self.Gst.State.NULL)
            self.pipeline = None
        if self.csv_writer is not None:
            self.csv_writer.close()


class GstreamerVideoSyncNode(Node):
    def __init__(self):
        super().__init__("gstreamer_video_sync")

        device = declare(self, "device", "/dev/video0", "V4L2 device to record.")
        output_dir = declare(
            self, "output_dir", "data/video", "Directory the video and sync CSV go to.")
        width = declare(self, "width", 1280, "Capture width.")
        height = declare(self, "height", 720, "Capture height.")
        fps = declare(self, "fps", 30, "Capture frame rate.")
        bitrate_kbps = declare(self, "bitrate_kbps", 5000, "Encoder bitrate, in kbit/s.")
        max_index_frames = declare(
            self, "max_index_frames", 300, "Frames kept in the sync index.")
        source_format = declare(
            self, "source_format", "mjpeg", "Camera output format: mjpeg or raw.")

        self.recorder = GStreamerUvcRecorder(
            device=device,
            output_dir=output_dir,
            width=width,
            height=height,
            fps=fps,
            bitrate_kbps=bitrate_kbps,
            max_index_frames=max_index_frames,
            source_format=source_format,
        )
        self.get_logger().info(f"Recording {device} to {self.recorder.video_path}")

        self.loop_index = 0

        self.frame_id_pub = self.create_publisher(Int64, "video/frame_id", 10)
        self.time_pub = self.create_publisher(Float64, "video/time_s", 10)
        self.delta_pub = self.create_publisher(Float64, "video/delta_ms", 10)
        self.path_pub = self.create_publisher(String, "video/path", 10)

        self.create_subscription(Int64, "loop/index", self.on_loop_index, 10)
        self.create_subscription(Int64, "loop/monotonic_ns", self.on_loop_tick, 10)

    def on_loop_index(self, msg):
        self.loop_index = msg.data

    def on_loop_tick(self, msg):
        loop_wall_time = datetime.datetime.now().isoformat(timespec="microseconds")
        sync_info = self.recorder.run(self.loop_index, msg.data, loop_wall_time)
        frame_id, pts_ns, monotonic_ns, video_time_s, delta_ms, video_path = sync_info

        self.path_pub.publish(String(data=str(video_path)))
        if frame_id == "":
            self.get_logger().warn("no recorded frame yet", throttle_duration_sec=5.0)
            return

        self.frame_id_pub.publish(Int64(data=int(frame_id)))
        self.time_pub.publish(Float64(data=float(video_time_s)))
        if delta_ms != "":
            self.delta_pub.publish(Float64(data=float(delta_ms)))

    def destroy_node(self):
        if self.recorder is not None:
            self.recorder.shutdown()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = GstreamerVideoSyncNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
