import math
import random
import time
import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure, iterate_structure
from scipy.spatial import cKDTree


def clip(val, low, high):
    return low if val < low else (high if val > high else val)

def wrap_angle(a):
    return math.atan2(math.sin(a), math.cos(a))

# Grow every occupied cell of grid
# Makes obstacles appear larger so planned path keeps kart body clear
def inflate(grid, resolution, max_radius):
    r = max(1, int(math.ceil(max_radius / resolution)))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    disk = (xx * xx + yy * yy) <= r * r
    return binary_dilation(grid, structure=disk)

## RRT Stuff
class RRTNode:
    # One node in the search tree
    def __init__(self, x, y, heading, steering, parent=None, traj=None):
        self.x = x
        self.y = y
        self.heading = heading
        self.steering = steering
        self.parent = parent
        self.traj = traj

## Basic RRT
class KartRRT:
    def __init__(
            self,
            grid,
            resolution=0.2, # meters per grid cell
            wheelbase=1,    # axle-to-axle distance (m)
            k_length=1.0,   # kart length (m)
            k_width=1.0,    # kart width (m)
            max_steer=30.0, # max steering angle (deg)
            max_steer_rate=90.0, # max steering speed (deg/s)
            speed=3.0, # speed for simulation (m/s)
            body_points=3, # how many points along body are collision checked
            edge_dt=0.3, # duration of each arc grown by steer_torward (s)
            sim_dt=0.08, # integration step (s) for bicycle model
            steer_fan=7, # number of discrete steering angles to try
            iters=180, # maximum number of sampling iterations
            sample_rate=0.45, # chance of sampling goal instead of a random point
            goal_xy_threshold=1.2, # how close (m) the final point must be to goal
            goal_yaw_threshold=math.radians(35), # how close (rad) heading must be to goal
            rng_seed=None,
    ):
        self.grid = grid
        self.resolution = resolution
        self.height, self.width = grid.shape

        self.wheelbase = wheelbase
        self.max_steer = math.radians(max_steer)
        self.max_steer_rate = math.radians(max_steer_rate)
        self.speed = speed

        self.kart_length = k_length
        self.kart_width = k_width
        self.body_points = body_points

        self.edge_dt = edge_dt
        self.sim_dt = sim_dt
        self.steer_candidates = np.linspace(-self.max_steer, self.max_steer, steer_fan)

        self.iters = iters
        self.sample_rate = sample_rate
        self.goal_xy_threshold = goal_xy_threshold
        self.goal_yaw_threshold = goal_yaw_threshold
        self.rng = random.Random(rng_seed)

        # Precompute list of free cell centers to sample from
        free = np.argwhere(~grid)
        if len(free):
            self.free_xy = np.column_stack([(free[:, 1] + 0.5) * resolution,
                                            (free[:, 0] + 0.5) * resolution])
        else:
            self.free_xy = None

        # Points along kart centerline at which collision is checked
        half_len = self.kart_length / 2.0
        self.body_offsets = np.linspace(-half_len, half_len, self.body_points)

    def point_free(self, x, y):
        px = math.floor(x / self.resolution)
        py = math.floor(y / self.resolution)
        if px < 0 or py < 0 or px >= self.width or py >= self.height:
            return False
        return not self.grid[py, px]
    
    def pose_free(self, x, y, heading):
        cos_h, sin_h = math.cos(heading), math.sin(heading)
        for offset in self.body_offsets:
            if not self.point_free(x + offset * cos_h, y + offset * sin_h):
                return False
        return True

    def edge_free(self, xs, ys, headings):
        return all(self.pose_free(x, y, h) for x, y, h in zip(xs, ys, headings))

    def simulate(self, x, y, heading, steering, target_steering, duration):
        xs, ys, headings, steerings = [x], [y], [heading], [steering]
        n_steps = max(1, int(round(duration / self.sim_dt)))
        current_steering = steering
        max_delta = self.max_steer_rate * self.sim_dt
 
        for _ in range(n_steps):
            delta = clip(target_steering - current_steering, -max_delta, max_delta)
            current_steering = clip(current_steering + delta, -self.max_steer, self.max_steer)
            x += self.speed * math.cos(heading) * self.sim_dt
            y += self.speed * math.sin(heading) * self.sim_dt
            heading = wrap_angle(heading + (self.speed / self.wheelbase) * math.tan(current_steering) * self.sim_dt)
            xs.append(x)
            ys.append(y)
            headings.append(heading)
            steerings.append(current_steering)
        return xs, ys, headings, steerings

    def steer_toward(self, nearest, sample):
        # Try to grow tree from nearest towards sample
        desired_heading = math.atan2(sample[1] - nearest.y, sample[0] - nearest.x)
        heading_error = wrap_angle(desired_heading - nearest.heading)
        target_steering = clip(heading_error, -self.max_steer, self.max_steer)
        candidate = min(self.steer_candidates, key=lambda s: abs(s - target_steering))
        xs, ys, headings, steerings = self.simulate(
            nearest.x, nearest.y, nearest.heading, nearest.steering, candidate, self.edge_dt)
 
        if not self.edge_free(xs, ys, headings):
            return None
        return RRTNode(xs[-1], ys[-1], headings[-1], steerings[-1], parent=nearest,
                       traj=(xs[1:], ys[1:], headings[1:], steerings[1:]))

    def try_reach_goal(self, node, goal):
        """Dedicated fine-grained simulation to connect directly to goal pose."""
        dist = math.hypot(goal[0] - node.x, goal[1] - node.y)
        if dist > 3.0:  # Only attempt when within 3 meters
            return None

        # Try multiple steering actions to curve into the goal orientation
        for steer in self.steer_candidates:
            duration = dist / self.speed
            xs, ys, headings, steerings = self.simulate(node.x, node.y, node.heading, node.steering, steer, duration)
            
            if not self.edge_free(xs, ys, headings):
                continue
                
            final_dist = math.hypot(xs[-1] - goal[0], ys[-1] - goal[1])
            h_err = abs(wrap_angle(headings[-1] - goal[2]))
            if final_dist < self.goal_xy_threshold and h_err < self.goal_yaw_threshold:
                return RRTNode(xs[-1], ys[-1], headings[-1], steerings[-1], parent=node,
                               traj=(xs[1:], ys[1:], headings[1:], steerings[1:]))
        return None
    
    def plan(self, start, goal, start_steer=0.0):
        """
        start: (x, y, heading) in grid metres, goal: (x, y, heading) in grid metres.
        Returns (path, info). path is an (M, 4) array of [x, y, heading, steering]
        sampled at sim_dt along the collision-checked arcs, or None if no path found.
        """
        t0 = time.perf_counter()
        root = RRTNode(start[0], start[1], start[2], start_steer)
        nodes = [root]
        node_coords = [(start[0], start[1])]
        goal_node = None
        it = 0
 
        for it in range(self.iters):
            if self.rng.random() < self.sample_rate:
                px, py = goal[0], goal[1]
            elif self.free_xy is not None:
                px, py = self.free_xy[self.rng.randrange(len(self.free_xy))]
            else:
                break
 
            tree = cKDTree(node_coords)
            _, idx = tree.query([px, py])
            nearest = nodes[idx]
 
            new_node = self.steer_toward(nearest, (px, py))
            if new_node is None:
                continue
 
            nodes.append(new_node)
            node_coords.append((new_node.x, new_node.y))
 
            direct = self.try_reach_goal(new_node, goal)
            if direct is not None:
                goal_node = direct
                nodes.append(goal_node)
                break
 
        info = {"time": time.perf_counter() - t0, "iterations": it + 1,
                "nodes": len(nodes), "found": goal_node is not None}
        if goal_node is None:
            return None, info
 
        chain = []
        cur = goal_node
        while cur.parent is not None:
            chain.append(cur.traj)
            cur = cur.parent
        chain.reverse()
 
        rows = [(root.x, root.y, root.heading, root.steering)]
        for xs, ys, hs, ss in chain:
            rows.extend(zip(xs, ys, hs, ss))
        return np.array(rows, dtype=float), info
 