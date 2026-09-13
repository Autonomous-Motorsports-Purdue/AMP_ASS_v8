import math
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, generate_binary_structure, iterate_structure

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
    def __init__(self, x, y, parent=None):
        self.x = x
        self.y = y
        self.parent = parent

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

    raw_grid, resolution = build_grid(obstacles, max_width=20.0, max_height=14.0, resolution=0.1)
    inflated = inflate(raw_grid, resolution, max_radius=0.55)
    start = (1.5, 2.0)
    goal = (12.5, 8.0)
    planner = RRT(inflated, resolution, step=0.5, iters=5000, sample_rate=0.1, goal_threshold=0.5, rng_seed=1)
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

    if path:
        xs, ys = zip(*path)
        ax.plot(xs, ys, color="tab:blue", linewidth=2.5, label="planned path")

    ax.plot(*start, "go", markersize=10, label="start")
    ax.plot(*goal, "r*", markersize=14, label="goal")
    ax.set_aspect("equal")
    ax.set_title(f"RRT on pseudo occupancy grid  ({info['time']*1000:.0f} ms, "
                 f"{info['nodes']} nodes)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

        