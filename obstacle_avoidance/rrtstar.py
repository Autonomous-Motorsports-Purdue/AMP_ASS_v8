import math
import random
import time
from rrt import (
    RRT,
    Node as BaseNode,
    RectObstacle,
    CircleObstacle,
    build_grid,
    inflate,
)


class Node(BaseNode):
    def __init__(self, x, y, heading=0.0, steering=0.0, parent=None):
        super().__init__(x, y, heading, steering, parent)
        self.cost = 0.0
        self.children = []

class RRTStar(RRT):
    def __init__(self, grid, resolution, step=0.5, search_radius=2.0, iters=5000, sample_rate=0.1, goal_threshold=0.5, rng_seed=None):
        super().__init__(grid, resolution, step, iters, sample_rate, goal_threshold, rng_seed)
        self.search_radius = search_radius

    def dist(self, n1, n2):
        return math.hypot(n1.x - n2.x, n1.y - n2.y)

    def plan(self, start, goal):
        t0 = time.perf_counter()
        
        # Initialize starting node
        start_node = Node(start[0], start[1])
        nodes = [start_node]
        goal_nodes = []
        iter_count = 0

        for iter_count in range(self.iters):
            # Random Sampling for direction, sometimes the rng selects the goal node to move towards the goal
            if self.rng_seed.random() < self.sample_rate:
                px, py = goal[0], goal[1]
            else:
                px = self.rng_seed.uniform(0, self.width * self.resolution)
                py = self.rng_seed.uniform(0, self.height * self.resolution)

            # Find Nearest Tree Node
            new_point = Node(px, py)
            nearest = min(nodes, key=lambda n: self.dist(n, new_point))

            # Steer
            theta = math.atan2(py - nearest.y, px - nearest.x)
            nx = nearest.x + self.step * math.cos(theta)
            ny = nearest.y + self.step * math.sin(theta)
            new_node = Node(nx, ny)

            if not self.segment_free(nearest.x, nearest.y, nx, ny):
                continue

            # Find Neighbors with search radius for optimization
            neighbors = [n for n in nodes if self.dist(n, new_node) <= self.search_radius]
            
            # Default to the nearest neighbor as the best parent of the node
            # calculate the cost to reach the new node from its proposed parent
            best_parent = nearest
            min_cost = nearest.cost + self.dist(nearest, new_node)

            # Choose Best Parent
            for neighbor in neighbors:
                potential_cost = neighbor.cost + self.dist(neighbor, new_node)
                
                if potential_cost < min_cost and self.segment_free(neighbor.x, neighbor.y, nx, ny):
                    best_parent = neighbor
                    min_cost = potential_cost

            new_node.parent = best_parent
            new_node.cost = min_cost
            best_parent.children.append(new_node)
            nodes.append(new_node)

            # Rewiring of Neighbors here
            for neighbor in neighbors:
                rewire_cost = new_node.cost + self.dist(new_node, neighbor)
                if rewire_cost < neighbor.cost and self.segment_free(nx, ny, neighbor.x, neighbor.y):
                    
                    if neighbor.parent:
                        neighbor.parent.children.remove(neighbor)
                    
                    neighbor.parent = new_node
                    new_node.children.append(neighbor)
                    
                    # Propagate the cost reduction down to all descendants recursively
                    self.propagate_cost(neighbor, rewire_cost - neighbor.cost)

            # Check Goal Condition
            if math.hypot(nx - goal[0], ny - goal[1]) < self.goal_threshold:
                goal_nodes.append(new_node)

        elapsed = time.perf_counter() - t0
        info = {"time": elapsed, "iterations": iter_count + 1, "nodes": len(nodes), "found": len(goal_nodes) > 0}

        if not goal_nodes:
            return None, nodes, info

        # Select the goal node with the lowest cost
        best_goal_node = min(goal_nodes, key=lambda n: n.cost)

        path = []
        current = best_goal_node
        while current is not None:
            path.append((current.x, current.y))
            current = current.parent
        path.reverse()

        return path, nodes, info

    def propagate_cost(self, node, cost_change):
        """Recursively updates the cost of all descendants when a parent is rewired."""
        node.cost += cost_change
        for child in node.children:
            self.propagate_cost(child, cost_change)

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, generate_binary_structure, iterate_structure
from scipy.spatial import cKDTree

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
    planner = RRTStar(inflated, resolution, rng_seed=1)
    path, nodes, info = planner.plan(start, goal)

    print(f"found: {info['found']}  time: {info['time']*1000:.1f} ms  "
          f"iterations: {info['iterations']}  tree size: {info['nodes']}")

    fig, ax = plt.subplots(figsize=(10, 7))

    extent = (
        0,
        raw_grid.shape[1] * resolution,
        0,
        raw_grid.shape[0] * resolution,
    )

    # Occupancy grid and inflated collision area
    ax.imshow(
        raw_grid,
        origin="lower",
        cmap="Greys",
        alpha=0.25,
        extent=extent,
    )

    ax.imshow(
        inflated,
        origin="lower",
        cmap="Reds",
        alpha=0.20,
        extent=extent,
    )

    # RRT* tree
    for node in nodes:
        if node.parent is None:
            continue

        ax.plot(
            [node.parent.x, node.x],
            [node.parent.y, node.y],
            color="steelblue",
            linewidth=0.5,
            alpha=0.45,
        )

    # Sampled nodes colored by cost
    node_x = [node.x for node in nodes]
    node_y = [node.y for node in nodes]
    node_costs = [node.cost for node in nodes]

    ax.scatter(
        node_x,
        node_y,
        c=node_costs,
        cmap="viridis",
        s=8,
        alpha=0.7,
        label="RRT* nodes",
    )

    # Final path
    if path is not None:
        path_x, path_y = zip(*path)

        ax.plot(
            path_x,
            path_y,
            color="orange",
            linewidth=3,
            label="optimized path",
        )

    ax.scatter(*start[:2], color="green", s=100, label="start", zorder=5)
    ax.scatter(*goal[:2], color="red", marker="*", s=160, label="goal", zorder=5)

    ax.set_title(
        f"RRT* tree — {info['nodes']} nodes, "
        f"{info['time'] * 1000:.1f} ms"
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

        