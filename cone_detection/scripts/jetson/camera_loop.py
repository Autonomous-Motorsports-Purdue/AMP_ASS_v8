#!/usr/bin/env python3
"""Live cone detection from a USB webcam on the Jetson, with honest timing.

    # inside the Ultralytics JetPack 6 container (see run_container.sh)
    python3 /cone/scripts/jetson/camera_loop.py --save /cone/run.mp4 --seconds 20   # boxes, to a video
    python3 /cone/scripts/jetson/camera_loop.py --show              # boxes on screen (needs GUI OpenCV)
    python3 /cone/scripts/jetson/camera_loop.py --seconds 30        # headless: time it, then summarise

Why not `yolo predict source=0`: that opens the camera at OpenCV's default,
often 640x480 YUYV, which shrinks distant cones below the size the model can
find. This asks for MJPG at the camera's full mode (Brio 105: 1920x1080 @ 30)
and prints what the camera actually delivered.

--imgsz must match the engine export_engine.sh built (default 736 1280 for a
16:9 camera). 1920x1080 then scales to 1280x720 - the same scale as the
square 1280 eval, so the bucketed recall in results/ still applies.
"""
import argparse
import collections
import os
import statistics
import sys
import threading
import time

import cv2


def open_camera(dev, w, h, fps):
    src = int(dev) if str(dev).isdigit() else dev
    cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
    if not cap.isOpened():
        sys.exit(f"could not open camera {dev} - check `ls /dev/video*`")
    # FOURCC must be set before the size: at 1080p, YUYV over USB usually
    # drops to ~5 fps, MJPG holds 30.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # newest frame, not a queued stale one
    gw, gh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    gfps = cap.get(cv2.CAP_PROP_FPS)
    four = int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode(errors="replace")
    print(f"camera: {gw}x{gh} @ {gfps:.0f} fps, {four}")
    if (gw, gh) != (w, h):
        print(f"[warn] asked for {w}x{h}, got {gw}x{gh} - small cones will suffer. "
              f"Check the modes with: v4l2-ctl -d {dev} --list-formats-ext")
    return cap


class Grabber:
    """Reads the camera on its own thread and keeps only the newest frame.

    Reading in the main loop serialises MJPG decode (~10-15 ms of CPU for
    1080p on an Orin Nano) with inference, so decode + model overran the 33 ms
    between frames and the loop fell to exactly half the camera rate (15 Hz
    with a 29 ms model). Here decode overlaps inference, and a stale frame is
    never processed when a newer one exists.
    """

    def __init__(self, cap):
        self.cap, self.cond = cap, threading.Condition()
        self.frame, self.t, self.idx, self.dead = None, 0.0, -1, False
        self.times = collections.deque(maxlen=60)
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def _run(self):
        while not self.dead:
            ok, f = self.cap.read()
            t = time.perf_counter()
            with self.cond:
                if not ok:
                    self.dead = True
                else:
                    self.frame, self.t, self.idx = f, t, self.idx + 1
                    self.times.append(t)
                self.cond.notify_all()

    def next(self, last_idx, timeout=2.0):
        """Newest frame with index > last_idx -> (frame, t_captured, idx)."""
        with self.cond:
            self.cond.wait_for(lambda: self.idx > last_idx or self.dead, timeout)
            if self.idx <= last_idx:
                return None, 0.0, last_idx
            return self.frame, self.t, self.idx

    def fps(self):
        with self.cond:
            ts = list(self.times)
        return (len(ts) - 1) / (ts[-1] - ts[0]) if len(ts) > 1 and ts[-1] > ts[0] else 0.0

    def stop(self):
        self.dead = True
        self.th.join(timeout=1.0)


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/cone/models/base_p2_1280.engine")
    ap.add_argument("--cam", default="/dev/video0")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--imgsz", type=int, nargs=2, default=[736, 1280],
                    metavar=("H", "W"), help="must match the engine")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--show", action="store_true",
                    help="draw boxes in a window - the Ultralytics container's OpenCV is headless, "
                         "so this usually needs --save instead")
    ap.add_argument("--save", default="", metavar="FILE.mp4",
                    help="write the frames with boxes to a video (half size). Costs CPU, so time "
                         "without it")
    ap.add_argument("--seconds", type=float, default=0, help="stop after N s (0 = until q / Ctrl+C)")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.model, task="detect")
    cap = open_camera(args.cam, args.width, args.height, args.fps)
    grab = Grabber(cap)

    # warm-up: the first TensorRT calls allocate and are not representative
    frame, _, _ = grab.next(-1)
    if frame is None:
        sys.exit("camera returned no frame")
    for _ in range(5):
        model.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)

    writer, save_size = None, (frame.shape[1] // 2, frame.shape[0] // 2)
    if args.save:
        args.save = os.path.abspath(args.save)
        if os.path.isdir("/cone") and not args.save.startswith("/cone/"):
            # only /cone is shared with the Jetson; anything else vanishes with the container
            print(f"[warn] {args.save} is inside the container only - it is NOT visible on the "
                  f"Jetson and is deleted when the container exits. Use --save /cone/run.mp4")
        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, save_size)
        if not writer.isOpened():
            sys.exit(f"could not open {args.save} for writing")
        print(f"saving boxes to {args.save} at {save_size[0]}x{save_size[1]} "
              f"(timing numbers are lower while saving)")

    model_ms, loop_ms, age_ms, counts = [], [], [], []
    last_idx, skipped = -1, 0
    t_start = t_prev = t_print = time.perf_counter()
    try:
        while True:
            frame, t_cap, idx = grab.next(last_idx)   # newest frame we haven't used
            if frame is None:
                print("camera stopped delivering frames")
                break
            if last_idx >= 0:
                skipped += idx - last_idx - 1          # frames the model was too slow for
            last_idx = idx
            t1 = time.perf_counter()
            r = model.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
            t2 = time.perf_counter()

            model_ms.append((t2 - t1) * 1e3)    # pre + inference + post, as bench_latency.py
            age_ms.append((t2 - t_cap) * 1e3)   # frame arrived -> boxes ready
            loop_ms.append((t2 - t_prev) * 1e3)
            counts.append(len(r.boxes))
            t_prev = t2

            if args.show or writer is not None:
                img = r.plot(line_width=2)
                hz = 1e3 / statistics.mean(loop_ms[-30:])
                cv2.putText(img, f"{hz:4.1f} Hz  model {model_ms[-1]:4.1f} ms  cones {counts[-1]}",
                            (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                if writer is not None:
                    writer.write(cv2.resize(img, save_size))
                if args.show:
                    try:
                        cv2.imshow("cones", img)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                    except cv2.error:
                        # opencv-python-headless: no window support at all
                        print("[warn] this OpenCV cannot open windows (headless build) - "
                              "continuing without --show. To see the boxes, record a video:\n"
                              "       python3 /cone/scripts/jetson/camera_loop.py "
                              "--save /cone/run.mp4 --seconds 20")
                        args.show = False

            if t2 - t_print >= 1.0:
                n = min(len(loop_ms), 30)
                print(f"{1e3 / statistics.mean(loop_ms[-n:]):5.1f} Hz | camera "
                      f"{grab.fps():4.1f} fps | model {statistics.median(model_ms[-n:]):5.1f} ms | "
                      f"frame->boxes {statistics.median(age_ms[-n:]):5.1f} ms | cones {counts[-1]}")
                t_print = t2
            if args.seconds and t2 - t_start >= args.seconds:
                break
    except KeyboardInterrupt:
        pass
    finally:
        grab.stop()
        cap.release()
        if writer is not None:
            writer.release()
            # run_container.sh passes the host folder it mounted at /cone
            host = os.environ.get("CONE_HOST_DIR", "<this folder>")
            shown = (host.rstrip("/") + "/" + args.save[len("/cone/"):]
                     if args.save.startswith("/cone/") else args.save)
            print(f"\nvideo saved: {args.save}  (on the Jetson: {shown})")
        if args.show:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

    if len(model_ms) > 1:
        cam = grab.fps()
        print("\n--- summary ---")
        print(f"frames        : {len(model_ms)} processed, {skipped} camera frames skipped")
        print(f"camera        : {cam:.1f} fps delivered (asked for {args.fps})"
              + ("  <- camera is slowing itself down, likely low light / auto-exposure"
                 if cam < 0.8 * args.fps else ""))
        print(f"loop rate     : {1e3 / statistics.mean(loop_ms[1:]):.1f} Hz")
        print(f"model / frame : median {statistics.median(model_ms):.1f} ms, "
              f"p95 {pct(model_ms, 95):.1f} ms  ->  model alone could do "
              f"{1e3 / statistics.median(model_ms):.0f} Hz")
        print(f"frame->boxes  : median {statistics.median(age_ms):.1f} ms, p95 {pct(age_ms, 95):.1f} ms")
        print(f"cones / frame : median {statistics.median(counts):.0f}")

if __name__ == "__main__":
    main()
