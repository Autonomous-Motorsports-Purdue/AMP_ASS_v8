import math
import random
import time
import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure, iterate_structure
from scipy.spatial import cKDTree


def clip(val, low, high):
    return low if val < low else (high if val > high else val)

def inflate(grid, resolution, max_radius):
    radius_cells = max(1, int(math.ceil(max_radius / resolution)))
    struct = iterate_structure(generate_binary_structure(2, 1), radius_cells)
    return binary_dilation(grid, structure=struct)

## RRT Stuff
class RRTNode:
    def __init__(self, x, y, heading, steering, parent=None):
        self.x = x
        self.y = y
        self.heading = heading
        self.steering = steering
        self.parent = parent

## Basic RRT
class KartRRT:
    def __init__(
            self,
            grid,
            resolution=0.2,
            wheelbase=1,
            k_length=1.0,
            k_width=1.0,
            max_steer=30.0,
            max_steer_rate=90.0,
            speed=3.0,
            body_points=3,
            edge_dt=0.3,
            sim_dt=0.08,
            steer_fan=7,
            iters=180,
            sample_rate=0.45,
            goal_xy_threshold=1.2,
            goal_yaw_threshold=math.radians(35),
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
        self.rng_seed = random.Random(rng_seed)

        half_len = self.kart_length / 2.0
        self.body_offsets = np.linspace(-half_len, half_len, self.body_points)

    def point_free(self, x, y):
        ## Check if point is outside of obstacle
        px, py = int(x / self.resolution), int(y / self.resolution)
        if px < 0 or py < 0 or px >= self.width or py >= self.height:
            return False
        return not self.grid[py, px]

    def pose_free(self, x, y, heading):
        cos_h, sin_h = math.cos(heading), math.sin(heading)
        for offset in self.body_offsets:
            px = x + offset * cos_h
            py = y + offset * sin_h
            if not self.point_free(px, py):
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
            heading += (self.speed / self.wheelbase) * math.tan(current_steering) * self.sim_dt
            heading = math.atan2(math.sin(heading), math.cos(heading))
            xs.append(x)
            ys.append(y)
            headings.append(heading)
            steerings.append(current_steering)
        return xs, ys, headings, steerings

    def steer_toward(self, nearest, sample):
        desired_heading = math.atan2(sample[1] - nearest.y, sample[0] - nearest.x)
        heading_error = math.atan2(math.sin(desired_heading - nearest.heading), math.cos(desired_heading - nearest.heading))
        target_steering = clip(heading_error, -self.max_steer, self.max_steer)
        candidate = min(self.steer_candidates, key=lambda x: abs(x - target_steering))
        xs, ys, headings, steerings = self.simulate(nearest.x, nearest.y, nearest.heading, nearest.steering, candidate, self.edge_dt)

        if not self.edge_free(xs, ys, headings):
            return None
        return RRTNode(xs[-1], ys[-1], headings[-1], steerings[-1], parent=nearest)

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
            h_err = abs(math.atan2(math.sin(headings[-1] - goal[2]), math.cos(headings[-1] - goal[2])))
            
            if final_dist < self.goal_xy_threshold and h_err < self.goal_yaw_threshold:
                return RRTNode(xs[-1], ys[-1], headings[-1], steerings[-1], parent=node)
        return None
    
    def plan(self, start, goal):
        t0 = time.perf_counter()
        nodes = [RRTNode(start[0], start[1], start[2], 0.0)]
        node_coords = [(start[0], start[1])]
        goal_node = None

        for iter in range(self.iters):
            if self.rng_seed.random() < self.sample_rate:
                px, py = goal[0], goal[1]
            else:
                px = self.rng_seed.uniform(0, self.width * self.resolution)
                py = self.rng_seed.uniform(0, self.height * self.resolution)

            tree = cKDTree(node_coords)
            _, idx = tree.query([px, py])
            nearest = nodes[idx]

            new_node = self.steer_toward(nearest, (px, py))
            if new_node is None:
                continue

            nodes.append(new_node)
            node_coords.append((new_node.x, new_node.y))

            direct_goal_node = self.try_reach_goal(new_node, goal)
            if direct_goal_node is not None:
                goal_node = direct_goal_node
                nodes.append(goal_node)
                break

        elapsed = time.perf_counter() - t0
        info = {"time": elapsed, "iterations": iter + 1, "nodes": len(nodes), "found": goal_node is not None}

        if goal_node is None:
            return None, nodes, info

        path = []
        current = goal_node
        while current is not None:
            path.append((current.x, current.y, current.heading, current.steering))
            current = current.parent
        path.reverse()
        return path, nodes, info