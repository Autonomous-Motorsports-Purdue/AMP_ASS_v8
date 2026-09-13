import velodyne_decoder as vd
import dpkt
import socket
import time
import threading
import open3d as o3d
import numpy as np
import cv2

class Lidar():
    def __init__(self, port=2368, resolution=0.1, grid_range=20, height_range=(-.25, 2)):
        # Setup the decoder
        self.config = vd.Config(min_range=0, max_range=grid_range)
        self.decoder = vd.StreamDecoder(self.config)

        # Setup the UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('', port))

        print(f"Listening for Velodyne data on port {port}...")
        
    def _create_occupancy_grid(self, points, resolution, grid_range, height_range):
        #Translates world coordinates to a 2D occupancy grid.
    
        # 1. Height filtering (Ignore ground and ceiling)
        mask_z = (points[:, 2] > height_range[0]) & (points[:, 2] < height_range[1])
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
        
        grid[grid_y, grid_x] = 1  # 1 for binary data (Occupied)

        return grid
    
    def run(self, resolution=0.1, grid_range=29, height_range=(-.25,2)):
        # Drain all buffered packets and use the latest complete scan.
        # The VLP-16 sends many UDP packets per revolution; we need to
        # consume them all to avoid falling behind and seeing a rotating slice.
        self.sock.setblocking(False)
        latest_points = None

        while True:
            try:
                data, address = self.sock.recvfrom(2048)
            except BlockingIOError:
                break  # No more packets in the buffer
            stamp = time.time()
            scan = self.decoder.decode(stamp, data)
            if scan is not None:
                pts = scan[1]
                if len(pts) > 0:
                    latest_points = pts

        self.sock.setblocking(True)

        if latest_points is None:
            return

        points = latest_points

        occ_grid = self._create_occupancy_grid(points, resolution, grid_range, height_range)
        #Returns 2D occupancy grid with 1 and 0 values, and the points data with x,y,z,intensity values for each point
        return occ_grid, points
       

    
