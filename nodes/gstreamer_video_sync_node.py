#!/usr/bin/env python3
# in: loop/index, loop/monotonic_ns (Int64), UVC camera | out: data/video mp4 + sync csv, video/frame_id (Int64), video/time_s, video/delta_ms (Float64), video/path (String)
import csv
import datetime
import os
import threading
import time
from collections import deque

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Int64, String

SYSTEM_GSTREAMER_PLUGIN_PATH = "/usr/lib/x86_64-linux-gnu/gstreamer-1.0"


def import_gst():
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


class GstreamerVideoSyncNode(Node):
    def __init__(self):
        super().__init__("gstreamer_video_sync")
        device = self.declare_parameter("device", "/dev/video0").value
        output_dir = self.declare_parameter("output_dir", "data/video").value
        width = self.declare_parameter("width", 1280).value
        height = self.declare_parameter("height", 720).value
        fps = self.declare_parameter("fps", 30).value
        source_format = self.declare_parameter("source_format", "mjpeg").value

        os.makedirs(output_dir, exist_ok=True)
        start_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.video_path = os.path.join(output_dir, f"{start_time}.mp4")

        self.csvfile = open(os.path.join(output_dir, f"{start_time}_sync.csv"), "w", newline="")
        self.csvwriter = csv.writer(self.csvfile)
        self.csvwriter.writerow([
            "loop_index", "loop_monotonic_ns", "loop_wall_time", "nearest_camera_frame_id",
            "nearest_frame_pts_ns", "nearest_frame_monotonic_ns", "video_time_s",
            "delta_loop_to_frame_ms", "video_path"])

        # (frame_id, pts_ns, monotonic_ns) of recent frames, filled from the GStreamer thread
        self.frames = deque(maxlen=300)
        self.lock = threading.Lock()
        self.frame_id = 0
        self.pts_to_monotonic_offset_ns = None
        self.loop_index = 0

        if source_format == "mjpeg":
            source = (f'v4l2src device="{device}" do-timestamp=true ! '
                      f"image/jpeg,width={width},height={height},framerate={fps}/1 ! jpegdec ! videoconvert !")
        else:
            source = (f'v4l2src device="{device}" do-timestamp=true ! '
                      f"video/x-raw,width={width},height={height},framerate={fps}/1 ! videoconvert !")
        # tee: one branch encodes to mp4, the other feeds an appsink so we can timestamp frames
        pipeline = (
            f"{source} tee name=t "
            "t. ! queue ! videoconvert ! "
            f"x264enc tune=zerolatency speed-preset=ultrafast bitrate=5000 key-int-max={fps} ! "
            f'h264parse ! mp4mux faststart=true ! filesink location="{os.path.abspath(self.video_path)}" '
            "t. ! queue leaky=downstream max-size-buffers=2 ! videoconvert ! video/x-raw,format=RGB ! "
            "appsink name=sync_sink emit-signals=true sync=false max-buffers=2 drop=true")

        self.Gst = import_gst()
        self.pipeline = self.Gst.parse_launch(pipeline)
        self.pipeline.get_by_name("sync_sink").connect("new-sample", self.on_new_sample)
        self.pipeline.set_state(self.Gst.State.PLAYING)

        self.create_subscription(Int64, "loop/index", self.on_loop_index, 10)
        self.create_subscription(Int64, "loop/monotonic_ns", self.run, 10)
        self.frame_id_pub = self.create_publisher(Int64, "video/frame_id", 10)
        self.time_pub = self.create_publisher(Float64, "video/time_s", 10)
        self.delta_pub = self.create_publisher(Float64, "video/delta_ms", 10)
        self.path_pub = self.create_publisher(String, "video/path", 10)

    def on_new_sample(self, sink):
        buffer = sink.emit("pull-sample").get_buffer()
        pts_ns = buffer.pts
        if pts_ns == self.Gst.CLOCK_TIME_NONE:
            return self.Gst.FlowReturn.OK
        now_ns = time.monotonic_ns()
        with self.lock:
            # map GStreamer pts onto the monotonic clock using the first frame
            if self.pts_to_monotonic_offset_ns is None:
                self.pts_to_monotonic_offset_ns = now_ns - pts_ns
            self.frame_id += 1
            self.frames.append((self.frame_id, pts_ns, pts_ns + self.pts_to_monotonic_offset_ns))
        return self.Gst.FlowReturn.OK

    def on_loop_index(self, msg):
        self.loop_index = msg.data

    def run(self, msg):
        loop_ns = msg.data
        loop_wall_time = datetime.datetime.now().isoformat(timespec="microseconds")
        self.path_pub.publish(String(data=self.video_path))
        with self.lock:
            if not self.frames:
                return
            frame_id, pts_ns, frame_ns = min(self.frames, key=lambda f: abs(f[2] - loop_ns))

        video_time_s = pts_ns / 1e9
        delta_ms = (loop_ns - frame_ns) / 1e6
        self.csvwriter.writerow([self.loop_index, loop_ns, loop_wall_time, frame_id, pts_ns,
                                 frame_ns, video_time_s, delta_ms, self.video_path])
        self.csvfile.flush()

        self.frame_id_pub.publish(Int64(data=frame_id))
        self.time_pub.publish(Float64(data=video_time_s))
        self.delta_pub.publish(Float64(data=delta_ms))

    def destroy_node(self):
        # EOS so mp4mux writes a playable file
        self.pipeline.send_event(self.Gst.Event.new_eos())
        self.pipeline.get_bus().timed_pop_filtered(
            2 * self.Gst.SECOND, self.Gst.MessageType.EOS | self.Gst.MessageType.ERROR)
        self.pipeline.set_state(self.Gst.State.NULL)
        self.csvfile.close()
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
