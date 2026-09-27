import math
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, generate_binary_structure, iterate_structure
from scipy.spatial import cKDTree


def clip(val, low, high):
    return low if val < low else (high if val > high else val)

class Obstacle:
    def mask(self, xx, yy):
        return np.zeros_like(xx, dtype=bool)

    def stamp(self, grid, xx, yy):
        grid |= self.mask(xx, yy)

class RectObstacle(Obstacle):
    def __init__(self, cx, cy, h_width, h_height):
        self.cx = cx
        self.cy = cy
        self.h_width = h_width
        self.h_height = h_height

    def mask(self, xx, yy):
        return ((np.abs(xx - self.cx) <= self.h_width) &
                (np.abs(yy - self.cy) <= self.h_height))

class CircleObstacle(Obstacle):
    def __init__(self, cx, cy, radius):
        self.cx = cx
        self.cy = cy
        self.radius = radius

    def mask(self, xx, yy):
        return (xx - self.cx) ** 2 + (yy - self.cy) ** 2 <= self.radius ** 2

def build_grid(obstacles, max_width=20.0, max_height=14.0, resolution=0.1, border=True):
    width, height = int(max_width / resolution), int(max_height / resolution)
    grid = np.zeros((height, width), dtype=bool)
    col_idx, row_idx = np.meshgrid(np.arange(width), np.arange(height))
    xx = col_idx * resolution
    yy = row_idx * resolution

    for obstacle in obstacles:
        obstacle.stamp(grid, xx, yy)

    if border:
        grid[0:2, :] = True
        grid[-2:, :] = True
        grid[:, 0:2] = True
        grid[:, -2:] = True

    return grid, resolution

def inflate(grid, resolution, max_radius):
    radius_cells = max(1, int(math.ceil(max_radius / resolution)))
    struct = iterate_structure(generate_binary_structure(2, 1), radius_cells)
    return binary_dilation(grid, structure=struct)

## RRT Stuff
class Node:
    def __init__(self, x, y, heading, steering, parent=None):
        self.x = x
        self.y = y
        self.heading = heading
        self.steering = steering
        self.parent = parent

## Basic RRT
class RRT:
    def __init__(self, grid, resolution, step=0.5, iters=5000, sample_rate=0.1, goal_threshold=0.5, rng_seed=None):
        self.grid = grid
        self.resolution = resolution
        self.step = step
        self.iters = iters
        self.sample_rate = sample_rate
        self.goal_threshold = goal_threshold
        self.rng_seed = random.Random(rng_seed)
        self.height, self.width = grid.shape

    def point_free(self, x, y):
        ## Check if point is outside of obstacle
        px, py = int(x / self.resolution), int(y / self.resolution)
        if px < 0 or py < 0 or px >= self.width or py >= self.height:
            return False
        return not self.grid[py, px]

    def segment_free(self, x0, y0, x1, y1):
        ## Check if entire segment is free
        distance = math.hypot(x1 - x0, y1 - y0)
        n = max(1, int(distance / (self.resolution * 2)))
        for x in range(n + 1):
            t = x / n
            if not self.point_free(x0 + t * (x1 - x0), y0 + t * (y1 - y0)):
                return False
        return True

    def plan(self, start, goal):
        ## Grow tree from start until goal is reached or iters time out
        t0 = time.perf_counter()
        nodes = [Node(*start)]
        goal_node = None
        iter = 0

        for iter in range(self.iters):
            if self.rng_seed.random() < self.sample_rate:
                px, py = goal
            else:
                px = self.rng_seed.uniform(0, self.width * self.resolution)
                py = self.rng_seed.uniform(0, self.height * self.resolution)

            ## Get nearest point
            nearest = min(nodes, key=lambda n: (n.x - px) ** 2 + (n.y - py) ** 2)

            theta = math.atan2(py - nearest.y, px - nearest.x)
            nx = nearest.x + self.step * math.cos(theta)
            ny = nearest.y + self.step * math.sin(theta)

            if not self.segment_free(nearest.x, nearest.y, nx, ny):
                continue

            new_node = Node(nx, ny, nearest)
            nodes.append(new_node)

            if math.hypot(nx - goal[0], ny - goal[1]) < self.goal_threshold:
                goal_node = new_node
                break

        elapsed = time.perf_counter() - t0
        info = {"time": elapsed, "iterations": iter + 1, "nodes": len(nodes), "found": goal_node is not None}

        if goal_node is None:
            return None, nodes, info

        path = []
        current = goal_node
        while current is not None:
            path.append((current.x, current.y))
            current = current.parent
        path.reverse()

        path = self.clean(path, attempts=80)
        return path, nodes, info

    def clean(self, path, attempts):
        ## Try to break down zig-zagging pattern, if straight line use it
        path = list(path)
        for _ in range(attempts):
            if len(path) < 3:
                break
            i, j = sorted(self.rng_seed.sample(range(len(path)), 2))
            if (j - i) < 2:
                continue
            if self.segment_free(*path[i], *path[j]):
                path = path[:i + 1] + path[j:]
        return path
    
## Kart RRT
class KartRRT(RRT):
    def __init__(self, grid, resolution, wheelbase=1.0, k_length=1.0, k_width=1.0,
                 max_steer=30.0, max_steer_rate=90.0, speed=2.0,
                 body_points=3, edge_dt=0.4, sim_dt=0.05,
                 steer_fan=15, iters=8000, sample_rate=0.25,
                 goal_xy_threshold=0.6, goal_yaw_threshold=math.radians(30),
                 rng_seed=None):
        self.grid = grid
        self.resolution= resolution
        self.height, self.width = grid.shape

        self.wheelbase = wheelbase
        self.max_steer = math.radians(max_steer)
        self.max_steer_rate = math.radians(max_steer_rate)
        self.speed = speed

        self.kart_length = k_length
        self.kart_width = k_width
        self.body_points = body_points ## How many points to check

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

    def pose_free(self, x, y, heading):
        cos_heading, sin_heading = math.cos(heading), math.sin(heading)
        for offset in self.body_offsets:
            px = x + offset * cos_heading
            py = y + offset * sin_heading
            if not self.point_free(px, py):
                return False
        return True

    def edge_free(self, xs, ys, headings):
        return all(self.pose_free(x, y, heading) for x, y, heading in zip(xs, ys, headings))

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
        return Node(xs[-1], ys[-1], headings[-1], steerings[-1], parent=nearest)

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
                return Node(xs[-1], ys[-1], headings[-1], steerings[-1], parent=node)
        return None
    
    def plan(self, start, goal):
        t0 = time.perf_counter()
        nodes = [Node(start[0], start[1], start[2], 0.0)]
        node_coords = [(start[0], start[1])]
        goal_node = None
        iter = 0

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

            # reached_xy = math.hypot(new_node.x - goal[0], new_node.y - goal[1]) < self.goal_xy_threshold
            # heading_error = math.atan2(math.sin(new_node.heading - goal[2]), math.cos(new_node.heading - goal[2]))
            # if reached_xy and abs(heading_error) < self.goal_yaw_threshold:
            #     goal_node = new_node
            #     break

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

        path = self.smooth(path, attempts=60)
        return path, nodes, info

    def smooth(self, path, attempts):
        path = list(path)
        for _ in range(attempts):
            if len(path) < 6:
                break
            i, j = sorted(self.rng_seed.sample(range(len(path)), 2))
            if (j - i) < 4:
                continue

            ax, ay, aheading, asteering = path[i]
            bx, by, _, _ = path[j]
            duration = self.edge_dt * max(1, (j - i) // max(1, int(self.edge_dt / self.sim_dt)))
            desired_heading = math.atan2(by - ay, bx - ax)
            heading_error = math.atan2(math.sin(desired_heading - aheading), math.cos(desired_heading - aheading))
            target_steering = clip(heading_error, -self.max_steer, self.max_steer)

            xs, ys, headings, steerings = self.simulate(ax, ay, aheading, asteering, target_steering, duration)
            reaches = math.hypot(xs[-1] - bx, ys[-1] - by) < self.goal_xy_threshold * 1.5
            if reaches and self.edge_free(xs, ys, headings):
                new_segment = list(zip(xs, ys, headings, steerings))
                path = path[:i] + new_segment + path[j:]
        return path



## Visualize (Heavily AI Warning)
def main():
    obstacles = [
        RectObstacle(cx=6.0, cy=4.0, h_width=0.6, h_height=1.5),
        RectObstacle(cx=6.0, cy=9.0, h_width=0.6, h_height=1.5),
        RectObstacle(cx=13.0, cy=3.0, h_width=1.5, h_height=0.5),
        RectObstacle(cx=13.0, cy=11.0, h_width=1.5, h_height=0.5),
        CircleObstacle(cx=10.0, cy=7.0, radius=1.2),
        CircleObstacle(cx=16.0, cy=7.0, radius=1.0)
    ]

    kart_width = 1.0
    raw_grid, resolution = build_grid(obstacles, max_width=20.0, max_height=14.0, resolution=0.1)
    inflated = inflate(raw_grid, resolution, max_radius=kart_width / 2 + 0.15)
    start = (1.5, 2.0, 2.0)
    goal = (17.5, 10.0, math.radians(90))
    ##planner = RRT(inflated, resolution, step=0.5, iters=5000, sample_rate=0.1, goal_threshold=0.5, rng_seed=1)
    planner = KartRRT(inflated, resolution, rng_seed=1)
    path, nodes, info = planner.plan(start, goal)

    print(f"found: {info['found']}  time: {info['time']*1000:.1f} ms  "
          f"iterations: {info['iterations']}  tree size: {info['nodes']}")

    fig, ax = plt.subplots(figsize=(8, 6))
    extent = (0, raw_grid.shape[1] * resolution, 0, raw_grid.shape[0] * resolution)
    ax.imshow(inflated, origin="lower", cmap="Greys", alpha=0.4, extent=extent)
    ax.imshow(raw_grid, origin="lower", cmap="Greys", extent=extent)

    for node in nodes:
        if node.parent is not None:
            ax.plot([node.x, node.parent.x], [node.y, node.parent.y], color="lightblue", linewidth=0.6)

    

    # if path:
    #     xs = [p[0] for p in path]
    #     ys = [p[1] for p in path]
    #     ax.plot(xs, ys, color = )

    # ax.plot(*start, "go", markersize=10, label="start")
    # ax.plot(*goal, "r*", markersize=14, label="goal")
    # ax.set_aspect("equal")
    # ax.set_title(f"RRT on pseudo occupancy grid  ({info['time']*1000:.0f} ms, "
    #              f"{info['nodes']} nodes)")
    # ax.legend(loc="upper left")
    # fig.tight_layout()
    # plt.show()

    if path:
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        ax.plot(xs, ys, color="tab:blue", linewidth=2.5, label="planned path")
        for p in path[::10]:
            x, y, yaw, steer = p
            ax.arrow(x, y, 0.4 * math.cos(yaw), 0.4 * math.sin(yaw),
                      head_width=0.15, color="tab:orange", alpha=0.8)

    ax.plot(*start[:2], "go", markersize=10, label="start")
    ax.plot(*goal[:2], "r*", markersize=14, label="goal")
    ax.set_aspect("equal")
    ax.set_title(f"Kart RRT ({info['time']*1000:.0f} ms, {info['nodes']} nodes)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    plt.show()

    if path:
        fig2, ax2 = plt.subplots(figsize=(8, 3))
        steers_deg = [math.degrees(p[3]) for p in path]
        ax2.plot(steers_deg)
        ax2.axhline(30, color="red", linestyle="--", linewidth=0.8)
        ax2.axhline(-30, color="red", linestyle="--", linewidth=0.8)
        ax2.set_xlabel("sample index")
        ax2.set_ylabel("steering angle (deg)")
        ax2.set_title("Steering angle along path (bounded to +/-30 deg)")
        fig2.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()

        