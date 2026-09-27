import velodyne_decoder as vd
import dpkt
import socket
import time
import open3d as o3d
import numpy as np
import cv2

# --- Testing ---
IS_LIVE = False                     # Set to False to read PCAP data from the file below
TEST_DATA_FILE = 'lidar_data.pcapng'   # Path to a PCAP file

# --- Config ---
GRID_RES = 0.1                      # Meters per cell
GRID_RANGE = 6                     # Range of grid in meters (Total ~140m square grid for our VLP 16)
HEIGHT_RANGE = (-.25, 2)            # Only include points between values in meters relative to the LiDAR
OBSTACLE_DILATE_KERNEL = 3          # (px/cells) merge nearby points into one blob
MIN_OBSTACLE_AREA = 3               # minimum blob area to count as an obstacle
MIN_OBSTACLE_DIM_M = 0.15           # minimum bounding box width/height to keep
DISPLAY = True
OUTPUT_VIDEO_PATH = "obstacle_map_output.mp4"
SAVE_VIDEO = False
SAVE_SAMPLE_FRAME = False
SAVE_FRAME_PATH = "sample_frame.png"
SAMPLE_FRAME_INDEX = 30

# --- Constants that should not change ---
PORT = 2368                         # Default port for Velodyne LiDAR sensors

def create_occupancy_grid(points, resolution, grid_range):
    """
    Translates world coordinates to a 2D occupancy grid.
    Formula: $index = \lfloor \frac{point + range}{resolution} \rfloor$
    """
    # 1. Height filtering (Ignore ground and ceiling)
    mask_z = (points[:, 2] > HEIGHT_RANGE[0]) & (points[:, 2] < HEIGHT_RANGE[1])
    points = points[mask_z]

    # 2. Spatial filtering (X and Y bounds)
    mask_xy = (np.abs(points[:, 0]) < grid_range) & (np.abs(points[:, 1]) < grid_range)
    points = points[mask_xy]

    # 3. Calculate Grid Size
    grid_size = int((grid_range * 2) / resolution)
    
    # 4. Map to indices (Shift by grid_range to keep values positive)
    grid_x = ((points[:, 0] + grid_range) / resolution).astype(np.int32)
    grid_y = ((points[:, 1] + grid_range) / resolution).astype(np.int32)

    # 5. Populate Grid
    grid = np.zeros((grid_size, grid_size), dtype=np.uint8)
    # Clip to avoid index errors at the very edge of the range
    grid_x = np.clip(grid_x, 0, grid_size - 1)
    grid_y = np.clip(grid_y, 0, grid_size - 1)
    
    grid[grid_y, grid_x] = 255  # 255 for high visibility (Occupied)
    
    return grid

def detect_obstacles(grid, dilate_kernel=OBSTACLE_DILATE_KERNEL, min_area=MIN_OBSTACLE_AREA):
    if dilate_kernel > 1:
        kernel = np.ones((dilate_kernel, dilate_kernel), np.uint8)
        working = cv2.dilate(grid, kernel, iterations=1)
    else:
        working = grid

    num_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(working, connectivity=8)

    obstacles = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        if max(w, h) * GRID_RES < MIN_OBSTACLE_DIM_M:
            continue
        cx, cy = centroids[i]
        obstacles.append({"x": x, "y": y, "w": w, "h": h, "area": int(area), "cx": cx, "cy": cy})
    return obstacles

def grid_to_world(px, py, resolution=GRID_RES, grid_range=GRID_RANGE):
    return (px * resolution - grid_range), (py * resolution - grid_range)


def create_point_cloud(points):
    """Standard O3D PointCloud creation with intensity coloring."""
    xyz = points[:, :3].copy()
    xyz[:, 2] = 0 # Flatten points to 2D plane for visualization
    intensity = points[:, 3]

    norm = intensity / intensity.max() if intensity.max() > 0 else intensity
    colors = np.zeros((len(norm), 3))
    colors[:, 0] = np.clip(1.5 - np.abs(4 * norm - 3), 0, 1) # R
    colors[:, 1] = np.clip(1.5 - np.abs(4 * norm - 2), 0, 1) # G
    colors[:, 2] = np.clip(1.5 - np.abs(4 * norm - 1), 0, 1) # B
    return xyz, colors

def render_occupancy_grid(grid, obstacles, display_size=(800, 800)):
    color_grid = cv2.cvtColor(grid, cv2.COLOR_GRAY2BGR)
    grid_size = grid.shape[0]

    for x in range(-GRID_RANGE, GRID_RANGE + 1):
        pix_pos = int((x + GRID_RANGE) / GRID_RES)
        cv2.line(color_grid, (pix_pos, 0), (pix_pos, grid.shape[0]), (40, 40, 40), 1)
        cv2.line(color_grid, (0, pix_pos), (grid.shape[1], pix_pos), (40, 40, 40), 1)

    origin_pix = int(GRID_RANGE / GRID_RES)
    cv2.line(color_grid, (origin_pix, 0), (origin_pix, grid.shape[0]), (100, 100, 100), 1)
    cv2.line(color_grid, (0, origin_pix), (grid.shape[1], origin_pix), (100, 100, 100), 1)

    color_grid[grid == 255] = [0, 0, 255]

    scale = display_size[0] / grid_size
    resized = cv2.resize(color_grid, display_size, interpolation=cv2.INTER_NEAREST)
    resized = cv2.flip(resized, 0)
    H = display_size[1]

    for obstacle in obstacles:
        x0 = int(obstacle["x"] * scale)
        y0 = int(obstacle["y"] * scale)
        ww = int(obstacle["w"] * scale)
        hh = int(obstacle["h"] * scale)
        y0_flipped = H - (y0 + hh)
        cv2.rectangle(resized, (x0, y0_flipped), (x0 + ww, y0_flipped + hh), (0, 255, 0), 2)
        w_m = obstacle["w"] * GRID_RES
        h_m = obstacle["h"] * GRID_RES
        wx, wy = grid_to_world(obstacle["cx"], obstacle["cy"])
        label = f"{w_m:.1f}x{h_m:.1f}m @({wx:.1f},{wy:.1f})"
        text_y = max(y0_flipped - 6, 12)
        cv2.putText(resized, label, (x0, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0), 1, cv2.LINE_AA)

    ox, oy_flipped = int(origin_pix * scale), H - int(origin_pix * scale)
    cv2.drawMarker(resized, (ox, oy_flipped), (255, 255, 0), cv2.MARKER_CROSS, 12, 2)
    return resized

## AI Stuff
class OutputSink:
    """Handles the optional GUI window and/or video file output."""
 
    def __init__(self, display, save_video, video_path, fps, frame_size):
        self.display = display
        self.writer = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(video_path, fourcc, fps, frame_size)
            if not self.writer.isOpened():
                print(f"WARNING: could not open video writer for {video_path}")
                self.writer = None
 
    def write(self, frame):
        if self.display:
            try:
                cv2.imshow("Occupancy Grid (with Scale)", frame)
                cv2.waitKey(1)
            except cv2.error as e:
                print(f"DISPLAY=True but no GUI backend is available ({e}); "
                      f"install non-headless opencv-python to use a live window.")
        if self.writer is not None:
            self.writer.write(frame)
 
    def close(self):
        if self.writer is not None:
            self.writer.release()
        if self.display:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

def read_live_data():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("", PORT))

    config = vd.Config(model=vd.Model.VLP16, min_range=0, max_range=GRID_RANGE)
    decoder = vd.StreamDecoder(config)

    sink = OutputSink(DISPLAY, SAVE_VIDEO, OUTPUT_VIDEO_PATH, 10, (800, 800))

    try:
        while True:
            data, _addr = sock.recvfrom(2048)
            stamp = time.time()
            scan = decoder.decode(stamp, bytearray(data))
            if scan is None:
                continue
            points = scan[1]
            if len(points) == 0:
                continue

            occ_grid = create_occupancy_grid(points, GRID_RES, GRID_RANGE)
            obstacles = detect_obstacles(occ_grid)
            frame = render_occupancy_grid(occ_grid, obstacles)
            sink.write(frame)
    finally:
        sink.close()
    


def read_from_file(file_path):
    config = vd.Config(model=vd.Model.VLP16, min_range=0, max_range=GRID_RANGE)
    decoder = vd.StreamDecoder(config)

    sink = OutputSink(DISPLAY, SAVE_VIDEO, OUTPUT_VIDEO_PATH, 10, (800, 800))

    scan_index = 0
    obstacle_counts = []
    
    with open(file_path, 'rb') as f:
        try:
            reader = dpkt.pcapng.Reader(f)
        except:
            f.seek(0)
            reader = dpkt.pcap.Reader(f)

        try:
            for stamp, buf in reader:
                try:
                    eth = dpkt.ethernet.Ethernet(buf)
                    if not isinstance(eth.data, dpkt.ip.IP):
                        continue
                    ip = eth.data
                    if not isinstance(ip.data, dpkt.udp.UDP):
                        continue
                    udp = ip.data
                    payload = bytearray(udp.data)
                    if len(payload) != 1206:
                        continue
                except Exception:
                    continue

                scan = decoder.decode(float(stamp), payload)
                if scan is None:
                    continue
                points = scan[1]
                if len(points) == 0:
                    continue

                occ_grid = create_occupancy_grid(points, GRID_RES, GRID_RANGE)
                obstacles = detect_obstacles(occ_grid)
                obstacle_counts.append(len(obstacles))

                frame = render_occupancy_grid(occ_grid, obstacles)
                sink.write(frame)

                if SAVE_SAMPLE_FRAME and scan_index == SAMPLE_FRAME_INDEX:
                    cv2.imwrite(SAVE_FRAME_PATH, frame)

                scan_index += 1
        finally:
            sink.close()
    if obstacle_counts:
        print(f"Processed {scan_index} scans. "
              f"Obstacles/scan -> min {min(obstacle_counts)}, "
              f"max {max(obstacle_counts)}, avg {sum(obstacle_counts)/len(obstacle_counts):.1f}")

if __name__ == "__main__":
    if IS_LIVE:
        read_live_data()
    else:
        read_from_file(TEST_DATA_FILE)