import numpy as np

MAZE_ROWS = 10
MAZE_COLS = 10
FROM = [[8, 1], [7, 8]]
TO = [1, 7]
OBSTACLES = [[[1, 1], [2, 1]],
             [[4, 4], [4, 5], [4, 6]],
             [[6, 5], [7, 5]]]
OBSTACLES_LINE = [1, 1], [2, 1], [4, 4], [4, 5], [4, 6], [6, 5], [7, 5]

EMPTY = 0
OBSTACLE = 1
GOAL = 2

maze = np.zeros((MAZE_ROWS, MAZE_COLS))
maze[0, 6] = GOAL
obstacle_loc = np.array(OBSTACLES_LINE) - 1
for o in obstacle_loc:
    maze[tuple(o)] = OBSTACLE


# grid = 20
# MAZE_ROWS = grid
# MAZE_COLS = grid
# maze = np.zeros((grid, grid))
# obstacle = np.random.random((grid, grid)) < 0.1
# maze[obstacle] = OBSTACLE
# maze[-1, -1] = GOAL
#
# FROM = [[0, 5]]
# TO = [9, 9]
