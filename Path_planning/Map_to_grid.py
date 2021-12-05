import numpy as np

#Creating the grid
from Path_planning.A_star_algorithm import run_A_Star

max_val = 10 # Size of the map

# Creating the occupancy grid
np.random.seed(0) # To guarantee the same outcome on all computers
data = np.random.rand(max_val, max_val) * 20 # Create a grid of 50 x 50 random values

# Converting the random values into occupied and free cells
limit = 15
occupancy_grid = data.copy()
occupancy_grid[data>limit] = 1
occupancy_grid[data<=limit] = 0

# occupancy_grid = [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
#                   [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
#                   [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
#                   [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
#                   [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
#                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                   [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
#                   [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
#                   [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
#                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

# occupancy_grid=np.array(occupancy_grid)


start = (0,0)
goal = (9,9)

path, visitedNodes = run_A_Star(occupancy_grid, start, goal)

print(path)
