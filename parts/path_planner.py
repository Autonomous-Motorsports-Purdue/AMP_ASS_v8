from ompl import base as om
from ompl import geometric as og
import math
from enum import Enum

#An Enum holding directions to simplify later code
class Direction(float, Enum):
    EAST = 0.0
    NORTH = math.pi / 2
    WEST = math.pi
    SOUTH = 3 * math.pi / 2
    
class PathPlanner():
    #initialize the class
    def __init__(self, turning_radius):
        #initialize the turning radius
        self.turning_radius = turning_radius
        self.occ_grid = None
        
    #(x, y, steering angle, heading)
    #plans out the path
    #end with a random point, will be given later
    def plan(self, occ_grid, st, en):
        #if occupancu grid doesn't exist, return 
        if occ_grid is None:
            return
        self.occ_grid = occ_grid
        
        # the DubinsStateSpace tracks X, Y, and Yaw, while being constrained by the turning radius
        space = om.DubinsStateSpace(self.turning_radius)
        
        #setup bounds for the x and y vars
        bounds = om.RealVectorBounds(2)
        bounds.setLow(0, 0.0)
        bounds.setHigh(0, float(len(occ_grid)))
        bounds.setLow(1, 0.0)
        bounds.setHigh(1, float(len(occ_grid[0])))
        space.setBounds(bounds)
        
        ss = og.SimpleSetup(space)
        
        stateInformation = ss.getSpaceInformation()
        #change checking resolution to 1% so that the algorithm checks for collisions more often, so the cart doesn't go through walls
        stateInformation.setStateValidityCheckingResolution(0.01)
        validity = valididty_checker(stateInformation, self.occ_grid)
        #check validity checker
        ss.setStateValidityChecker(validity)
        
        
        #assume start is (0,0)
        start = space.allocState()
        start.setX(float(st[0]))
        start.setY(float(st[1]))
        start.setYaw(float(st[2])) if len(st) > 2 else start.setYaw(0.0)
        
        end = space.allocState()
        end.setX(float(en[0]))
        end.setY(float(en[1]))
        end.setYaw(float(en[2])) if len(en) > 2 else end.setYaw(0.0)
        
        ss.setStartAndGoalStates(start, end)
        
        obj = om.PathLengthOptimizationObjective(stateInformation)
        ss.setOptimizationObjective(obj)
        
        #setup path planner to make a path, it uses RRT to make a path
        planner = og.RRTstar(stateInformation)
        ss.setPlanner(planner)
        
        #generate path, and pass in a time value to solve it in
        solved = ss.solve(1.0)
        #return path
        if solved:
            path = ss.getSolutionPath()
            return path
        else:
            print("No path possible")
            return None
class valididty_checker(om.StateValidityChecker):
    def __init__(self, spaceInfo, occ_grid, clearance_buffer = 0.3):
        super().__init__(spaceInfo)
        self.occ_grid = occ_grid
        self.clearance_buffer = clearance_buffer
        
    def isValid(self, state):
        if self.occ_grid is None:
            return True
        
        #get the x and y variables
        x = state.getX()
        y = state.getY()
        
        #if certain conditions are met, the data is invalid
        if x - self.clearance_buffer < 0.0 or x + self.clearance_buffer >= len(self.occ_grid) or y - self.clearance_buffer < 0.0 or y + self.clearance_buffer >= len(self.occ_grid[0]):
            return False
        
        min_x = int(x - self.clearance_buffer)
        max_x = int(x + self.clearance_buffer)
        min_y = int(y - self.clearance_buffer)
        max_y = int(y + self.clearance_buffer)
        
        for i in range(min_x, max_x + 1):
            for j in range(min_y, max_y + 1):
                if self.occ_grid[i][j] == 1:
                    return False
        return True
    
def test_planner():
    grid_size = 10
    occ_grid = [[0 for _ in range(grid_size)] for _ in range(grid_size)]
    
    for i in range(2, 4):
        for j in range(2, 5):
            occ_grid[i][j] = 1
    
    for i in range(6, 7):
        for j in range(4, 7):
            occ_grid[i][j] = 1
    for i in range(2, 4):
        for j in range(2, 5):
            occ_grid[i][j] = 1
    
    for i in range(4, 5):
        for j in range(4, 5):
            occ_grid[i][j] = 1
    for i in range(2, 4):
        for j in range(2, 5):
            occ_grid[i][j] = 1
    
    for i in range(0, 1):
        for j in range(5, 6):
            occ_grid[i][j] = 1
            
    for i in range(2, 4):
        for j in range(2, 5):
            occ_grid[i][j] = 1
    
    for i in range(3, 4):
        for j in range(7, 8):
            occ_grid[i][j] = 1
    for i in range(6, 7):
        for j in range(2, 3):
            occ_grid[i][j] = 1
    for i in range(7, 9):
        for j in range(7, 8):
            occ_grid[i][j] = 1

    for i in range(0, grid_size):
        for j in range(0, grid_size):
            print(str(occ_grid[i][j]), end=",")
        print("\n")
    planner = PathPlanner(turning_radius=1)
    
    # Define Start and Goal (x, y, heading)
    # Headings for the start and goal must be passed in radians, 0, pi/2, pi, 3pi/2
    start_pose = (1.5, 1.5, Direction.EAST)
    goal_pose = (8.5, 9.5, Direction.NORTH)
    
    print(f"Planning path from {start_pose} to {goal_pose}...")
    
    path = planner.plan(occ_grid, start_pose, goal_pose)
    
    visualize_path(
        occ_grid=occ_grid,
        path=path,
        start_pose=start_pose,
        goal_pose=goal_pose,
        num_states=300,
    )
    
    if path is not None:
        print("\nPath found! Waypoints:")
        
        # Interpolate the path to get evenly spaced points rather than just sparse joints
        # The number represents the total number of waypoints you want
        path.interpolate(20) 
        
        states = path.getStates()
        for i, state in enumerate(states):
            x = state.getX()
            y = state.getY()
            yaw = state.getYaw()
            print(f"Step {i:02d}: x={x:5.2f}, y={y:5.2f}, heading={yaw:5.2f} rad")
    else:
        print("Planner failed to find a valid path.")

#VISUALIZER CODE HERE:
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from path_planner import PathPlanner


def visualize_path(occ_grid, path, start_pose, goal_pose, num_states=150):
    if path is None:
        print("Cannot visualize: planner did not find a path.")
        return

    # Create evenly spaced states along the OMPL path.
    path.interpolate(num_states)
    states = path.getStates()

    x = np.array([state.getX() for state in states])
    y = np.array([state.getY() for state in states])
    yaw = np.array([state.getYaw() for state in states])

    grid = np.asarray(occ_grid)

    fig, ax = plt.subplots(figsize=(8, 8))
    height, width = grid.shape
    # occ_grid[x][y] is used by the planner, so transpose it for imshow:
    # image rows correspond to y and columns correspond to x.
    ax.imshow(
        grid.T,
        origin="lower",
        cmap="gray_r",
        interpolation="nearest",
        extent=[
                0.0,
                width,
                0.0,
                height,
            ],
    )

    # Full planned path.
    path_line, = ax.plot(
        [],
        [],
        color="tab:blue",
        linewidth=2,
        label="planned path",
    )

    # Portion of path already traveled.
    traveled_line, = ax.plot(
        [],
        [],
        color="tab:green",
        linewidth=3,
        label="traveled",
    )

    # Current vehicle position.
    vehicle, = ax.plot(
        [],
        [],
        marker="o",
        markersize=8,
        color="tab:red",
        label="vehicle",
    )

    # A short line showing the vehicle heading.
    heading_line, = ax.plot(
        [],
        [],
        color="tab:red",
        linewidth=2,
    )

    ax.plot(
        start_pose[0],
        start_pose[1],
        marker="o",
        markersize=8,
        color="tab:orange",
        label="start",
    )

    ax.plot(
        goal_pose[0],
        goal_pose[1],
        marker="*",
        markersize=8,
        color="tab:purple",
        label="goal",
    )

    ax.set_xlim(0.0, width)
    ax.set_ylim(0.0, height)
    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Path Planner Visualization")
    ax.grid(True, alpha=0.3)
    ax.legend()

    path_line.set_data(x, y)

    def update(frame):
        current_x = x[frame]
        current_y = y[frame]
        current_yaw = yaw[frame]

        traveled_line.set_data(x[: frame + 1], y[: frame + 1])
        vehicle.set_data([current_x], [current_y])

        heading_length = 0.5
        heading_line.set_data(
            [
                current_x,
                current_x + heading_length * math.cos(current_yaw),
            ],
            [
                current_y,
                current_y + heading_length * math.sin(current_yaw),
            ],
        )

        ax.set_title(
            f"Path Planner Visualization "
            f"({frame + 1}/{len(states)})"
        )

        return traveled_line, vehicle, heading_line

    animation = FuncAnimation(
        fig,
        update,
        frames=len(states),
        interval=50,
        blit=False,
        repeat=False,
    )

    animation.save(
        "planned_path.gif",
        writer="pillow",
        fps=20,
    )

    print("Saved animation to planned_path.gif")
    plt.show()
    
if __name__ == "__main__":
    test_planner()
        
        
    
        