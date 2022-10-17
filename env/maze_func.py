import numpy as np

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
move = np.array([[-1, 0], [1, 0], [0, -1], [0, 1], [0, 0]])


def get_avail_action(maze, loc):
    if maze[tuple(loc)] == 2:
        return [4]
    else:
        in_bound = []
        size = len(maze)
        if loc[0] != 0:
            in_bound.append(UP)
        if loc[0] != size - 1:
            in_bound.append(DOWN)
        if loc[1] != 0:
            in_bound.append(LEFT)
        if loc[1] != size - 1:
            in_bound.append(RIGHT)
        in_bound.append(4)

        avail_actions = []
        for a in in_bound:
            new_loc = loc + move[a]
            if maze[tuple(new_loc)] != 1:
                avail_actions.append(a)
        return avail_actions


def transition_loc(loc, action):
    return loc + move[action]


def terminated(goal, loc):
    return tuple(goal) == tuple(loc)
