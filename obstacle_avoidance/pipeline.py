import math 
import numpy as np
from scipy.spatial import cKDTree
from kartRRT import KartRRT, inflate
from pure_pursuit_obstacle_controller import PurePursuitController

# Convert occupancy mask from LiDAR grid into sensor-frame points
def mask_to_sensor_points(mask, resolution, grid_range):
    # mask : 2-D boolearn array
    # resolution : meters per cell
    # grid_range : half_width of grid in meters

    rows, cols = np.nonzero(mask)
    return np.column_stack([cols * resolution - grid_range,
                            rows * resolution - grid_range])

def densify_loop(x, y, step):
    # Resample points so consecutive points are one step apart
    xc, yc = np.append(x, x[0]), np.append(y, y[0])
    s = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(xc), np.diff(yc)))))
    n = max(2, int(s[-1] / step))
    t = np.linspace(0.0, s[-1], n, endpoint=False)
    return np.interp(t, s, xc), np.interp(t, s, yc)

class ObstacleAwarePursuit:
    def __init__(
            self,
            path_csv,
            controller_kwargs=None,  # passed to PurePursuitController
            rrt_kwargs=None,         # overrides for KartRRT
            kart_half_width=0.5,     # kart half-width for inflating obstacles (m)
            safety_margin=0.3,       # extra clearence around obstacle (m)
            track_half_width=2.5,    # driveable half-width of track (m)
            lidar_offset=(0.0, 0.0), # LiDAR position relative to kart center (m)
            lidar_yaw_offset=0.0,    # LiDAR mounting angle relative to the kart (rad)
            map_resolution=0.1,      # cell size of the track-frame map (m)
            obstacle_ttl_s=1.0,      # how long an obstacle an obstacle stays on map after it was last seen (s)
            check_horizon_m=6.0,     # how far ahead along the path to check for obstacles (m)
            self_exclusion_m=0.4,    # ignore LiDAR points closer than this (m)
            goal_clearence_m=3.0,    # goal must be at least this far past obstacle (m)
            min_goal_ahead_m=4.0,    # goal must be at least this far ahead of the kart (m)
            rrt_margin_m=4.0,        # extra room around the planning window (m)
            rrt_attempts=3,          # number of RRT tries before giving up
            map_pad_m=3.0,           # extra margin around the lap in the map (m)
    ):  
        # Controller that follows the path
        self.ctrl = PurePursuitController(path_csv, **(controller_kwargs or {}))

        # Copy of reference lap, detours are built from this
        self.ref_x = self.ctrl.path_x_m.copy()
        self.ref_y = self.ctrl.path_y_m.copy()
        self.ref_psi = self.ctrl.path_psi_rad.copy()
        self.ref_erpm = self.ctrl.target_erpm.copy()
        self.ref_s = self.ctrl.s_m.copy()
        self.ref_len = self.ref_s[-1] + math.hypot(self.ref_x[0] - self.ref_x[-1],
                                                   self.ref_y[0] - self.ref_y[-1])
        self.ref_tree = cKDTree(np.column_stack([self.ref_x, self.ref_y]))

        # Path currently being followed
        self.active = dict(x=self.ref_x.copy(), y=self.ref_y.copy(), psi=self.ref_psi.copy(), erpm=self.ref_erpm.copy())
        self.res = map_resolution

        # Obstacles are grown by this radius so a point path is enough to stay clear
        self.r_inf = kart_half_width + safety_margin
        self.ttl = obstacle_ttl_s
        self.check_horizon = check_horizon_m
        self.self_excl = self_exclusion_m
        self.goal_clearence = goal_clearence_m
        self.min_goal_ahead = min_goal_ahead_m
        self.rrt_margin = rrt_margin_m
        self.rrt_attempts = rrt_attempts
        self.lidar_offset = lidar_offset
        self.lidar_yaw_offset = lidar_yaw_offset

        # Setting for each RRT run
        self.rrt_kwargs = dict(wheelbase=1.0, k_length=1.0, k_width=1.0, speed=3.0, iters=800,
                               goal_xy_threshold=0.6, goal_yaw_threshold=math.radians(25), sample_rate=0.4)
        self.rrt_kwargs.update(rrt_kwargs or {})

        # Build the track-frame map
        xmin = self.ref_x.min() - track_half_width - map_pad_m
        xmax = self.ref_x.max() + track_half_width + map_pad_m
        ymin = self.ref_y.min() - track_half_width - map_pad_m
        ymax = self.ref_y.max() + track_half_width + map_pad_m
        self.ox, self.oy = xmin, ymin
        self.H = int(math.ceil((ymax - ymin) / self.res))
        self.W = int(math.ceil((xmax - xmin) / self.res))

        # Static walls
        dx, dy = densify_loop(self.ref_x, self.ref_y, 0.05)
        xs = self.ox + (np.arange(self.W) + 0.5) * self.res
        ys = self.oy + (np.arange(self.H) + 0.5) * self.res
        X, Y = np.meshgrid(xs, ys)
        dist, _ = cKDTree(np.column_stack([dx, dy])).query(np.column_stack([X.ravel(), Y.ravel()]))
        in_track = (dist <= track_half_width).reshape(self.H, self.W)
        self.static_blocked = inflate(~in_track, self.res, self.r_inf)

        # Disk offsets to grow individual obstacle cells
        r = max(1, int(math.ceil(self.r_inf / self.res)))
        dr, dc = np.mgrid[-r:r + 1, -r:r + 1]
        keep = (dr * dr + dc * dc) <= r * r
        self.disk_dr, self.disk_dc = dr[keep], dc[keep]

        # Time each cell was last seen by LiDAR
        self.last_seen = np.full((self.H, self.W), -np.inf)
        self.blocked = self.static_blocked.copy()

    # Convert (x, y) into map (row, col) indices
    def _cell(self, x, y):
        return (int(math.floor((y - self.oy) / self.res)),
                int(math.floor((x - self.ox) / self.res)))

    # Check if blocked (x, y)
    def _blocked_at(self, x, y):
        r, c = self._cell(x, y)
        if r < 0 or c < 0 or r >= self.H or c >= self.W:
            return True
        return bool(self.blocked[r, c])

    # Add scans obstacle points to the map
    def update_map(self, x, y, yaw, lidar_pts, now):

        # Find LiDAR on track frame
        c, s = math.cos(yaw), math.sin(yaw)
        lx_off, ly_off = self.lidar_offset
        lx = x + c * lx_off - s * ly_off
        ly = y + s * lx_off + c * ly_off
        h = yaw + self.lidar_yaw_offset
        ch, sh = math.cos(h), math.sin(h)

        if len(lidar_pts):
            # Crop outside points and translate into track frame
            pts = lidar_pts[np.hypot(lidar_pts[:, 0], lidar_pts[:, 1]) > self.self_excl]
            wx = lx + ch * pts[:, 0] - sh * pts[:, 1]
            wy = ly + sh * pts[:, 0] + ch * pts[:, 1]
            rows = np.floor((wy - self.oy) / self.res).astype(int)
            cols = np.floor((wx - self.ox) / self.res).astype(int)
            keep = (rows >= 0) & (rows < self.H) & (cols >= 0) & (cols < self.W)
            self.last_seen[rows[keep], cols[keep]] = now

        # Grow obstacle cells and update blocked grid
        fr, fc = np.nonzero((now - self.last_seen) < self.ttl)
        blocked = self.static_blocked.copy()
        if len(fr):
            R = (fr[:, None] + self.disk_dr[None, :]).ravel()
            C = (fc[:, None] + self.disk_dc[None, :]).ravel()
            ok = (R >= 0) & (R < self.H) & (C >= 0) & (C < self.W)
            blocked[R[ok], C[ok]] = True
        self.blocked = blocked

    # Find furthest blocked point on path within horizon
    def _last_blocked_ahead(self, path, idx, horizon):
        x, y = path["x"], path["y"]
        n = len(x)
        last, dist = None, 0.0
        for k in range(n):
            j = (idx + k) % n
            if k:
                p = (j - 1) % n
                dist += math.hypot(x[j] - x[p], y[j] - y[p])
                if dist > horizon:
                    break
            if self._blocked_at(x[j], y[j]):
                last = j
        return last

    # Build detour around obstacle and update path
    def _replan(self, x, y, yaw, blocked_idx):
        n = len(self.ref_x)
        bx, by = self.active["x"][blocked_idx], self.active["y"][blocked_idx]
        j0 = int(self.ref_tree.query([bx, by])[1])
        r0 = int(self.ref_tree.query([x, y])[1])

        # First unblocked reference point that is far enough past obstacle and kart
        goal = None
        for k in range(n):
            j = (j0 + k) % n
            past_obstacle = (self.ref_s[j] - self.ref_s[j0]) % self.ref_len
            from_kart = (self.ref_s[j] - self.ref_s[r0]) % self.ref_len
            if past_obstacle < self.goal_clearence or from_kart < self.min_goal_ahead:
                continue
            if self._blocked_at(self.ref_x[j], self.ref_y[j]):
                continue
            goal = j
            break
        if goal is None:
            return False

        gx, gy, gh = self.ref_x[goal], self.ref_y[goal], self.ref_psi[goal]

        # Planning window containing kart, goal, and reference between them
        seg = [(r0 + k) % n for k in range(((goal - r0) % n) + 1)]
        xs = np.concatenate([[x, gx], self.ref_x[seg]])
        ys = np.concatenate([[y, gy], self.ref_y[seg]])
        m = self.rrt_margin
        c0 = max(0, int(math.floor((xs.min() - m - self.ox) / self.res)))
        c1 = min(self.W, int(math.ceil((xs.max() + m - self.ox) / self.res)))
        r0_ = max(0, int(math.floor((ys.min() - m - self.oy) / self.res)))
        r1 = min(self.H, int(math.ceil((ys.max() + m - self.oy) / self.res)))
        sub = self.blocked[r0_:r1, c0:c1]
        lox, loy = self.ox + c0 * self.res, self.oy + r0_ * self.res

        start_local = (x - lox, y - loy, yaw)
        goal_local = (gx - lox, gy - loy, gh)

        # Run RRT with random seed
        path = None
        for attempt in range(self.rrt_attempts):
            planner = KartRRT(sub, resolution=self.res, rng_seed=attempt, **self.rrt_kwargs)
            path, _info = planner.plan(start_local, goal_local)
            if path is not None:
                break
        if path is None:
            return False

        # Convert detour to track coordinates
        det_x = path[:, 0] + lox
        det_y = path[:, 1] + loy
        det_psi = path[:, 2]

        # Integrate detour into path
        tail = [(goal + k) % n for k in range(((r0 - goal) % n))]
        new_x = np.concatenate([det_x, self.ref_x[tail]])
        new_y = np.concatenate([det_y, self.ref_y[tail]])
        new_psi = np.concatenate([det_psi, self.ref_psi[tail]])
        new_erpm = np.concatenate([np.full(len(det_x), self.ref_erpm[goal]), self.ref_erpm[tail]])
        self.active = dict(x=new_x, y=new_y, psi=new_psi, erpm=new_erpm)
        self.ctrl.set_path(new_x, new_y, new_psi, new_erpm)
        return True

    # Run one control tick
    def step(self, x, y, yaw_deg, speed_mps, lidar_pts, now):
        # x, y : kart position (m)
        # yaw_deg : kart heading (deg)
        # speed_mps : kart speed (m/s)
        # lidar_points: (N, 2) obstacle points in sensor frame (m)
        # now : current time (s)

        yaw = math.radians(yaw_deg)

        # Update obstacle map with scan
        self.update_map(x, y, yaw, lidar_pts, now)

        # Chekc path ahead for blocked points
        self.ctrl.find_closest_idx_forward(x, y)
        blocked_idx = self._last_blocked_ahead(self.active, self.ctrl.closest_idx, self.check_horizon)
        status = "follow"
        if blocked_idx is not None:
            # If blocked look for detour
            if self._replan(x, y, yaw, blocked_idx):
                status = "replanned"
            else:
                return 0, 0.0, "blocked_no_path"

        # Normal pure pursuit command for new path
        erpm, steer, _ = self.ctrl.run(x, y, yaw_deg, speed_mps)
        return erpm, steer, status
