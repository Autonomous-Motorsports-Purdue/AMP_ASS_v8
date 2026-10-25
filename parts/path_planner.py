from ompl import base as om
from ompl import geometric as og
import math

class PathPlanner():
    #initialize the class
    def __init__(self, turning_radius = 1.0):
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
        bounds.setHigh(0, float(len(occ_grid) - 1))
        bounds.setLow(1, 0.0)
        bounds.setHigh(1, float(len(occ_grid[0]) - 1))
        space.setBounds(bounds)
        
        ss = og.SimpleSetup(space)
        
        stateInformation = ss.getSpaceInformation()
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
        
        #setup path planner to make a path, it uses RRT to make a path
        planner = og.RRT(stateInformation)
        ss.setPlanner(planner)
        
        #generate path, and pass in a time value to solve it in
        solved = ss.solve(3.0)
        #return path
        if solved:
            ss.simplifySolution()
            path = ss.getSolutionPath()
            return path
        else:
            print("No path possible")
            return None
class valididty_checker(om.StateValidityChecker):
    def __init__(self, spaceInfo, occ_grid):
        super().__init__(spaceInfo)
        self.occ_grid = occ_grid
        
    def isValid(self, state):
        if self.occ_grid is None:
            return True
        
        #get the x and y variables
        x = int(state.getX())
        y = int(state.getY())
        
        #if certain conditions are met, the data is invalid
        if x < 0 or x >= len(self.occ_grid) or y < 0 or y >= len(self.occ_grid[0]):
            return False
        
        return (self.occ_grid[x][y] == 0)
    
def test_planner():
    grid_size = 10
    occ_grid = [[0 for _ in range(grid_size)] for _ in range(grid_size)]
    
    for i in range(4, 7):
        for j in range(2, 8):
            occ_grid[i][j] = 1

    planner = PathPlanner(turning_radius=0.5)
    
    # Define Start and Goal (x, y, heading)
    # Heading 0.0 is pointing East (along positive X axis)
    # Heading pi/2 is pointing North (along positive Y axis)
    start_pose = (1, 1, 0.0) 
    goal_pose = (8, 8, math.pi / 2) 
    
    print(f"Planning path from {start_pose} to {goal_pose}...")
    
    path = planner.plan(occ_grid, start_pose, goal_pose)
    
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

if __name__ == "__main__":
    test_planner()
        
        
    
        