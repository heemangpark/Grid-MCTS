import numpy as np
from config import TO

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
move = np.array([[0, 1], [0, -1], [-1, 0], [1, 0]])


def avail_action(maze, loc):
    temp_avail_actions = []
    map_x, map_y = maze.shape
    if loc[1] != map_y - 1:
        temp_avail_actions.append(UP)
    if loc[1] != 0:
        temp_avail_actions.append(DOWN)
    if loc[0] != map_x - 1:
        temp_avail_actions.append(RIGHT)
    if loc[0] != 0:
        temp_avail_actions.append(LEFT)

    avail_actions = []
    for a in temp_avail_actions:
        new_loc = loc + move[a]
        if not maze[tuple(new_loc)] == 1:
            avail_actions.append(a)

    return avail_actions


def transition_loc(loc, action):
    return loc + move[action]


def terminated(maze, loc):
    return maze[tuple(loc)] == TO