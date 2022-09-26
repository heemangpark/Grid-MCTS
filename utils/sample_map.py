import numpy as np


def grid(args):
    maze = np.zeros((args['size'], args['size']))
    obstacle = np.random.random((args['size'], args['size'])) < args['difficulty']
    maze[obstacle] = args['cell_type']['obstacle']

    rand_loc = np.random.choice(args['size'], 4)
    if rand_loc[0] == rand_loc[2] and rand_loc[1] == rand_loc[3]:
        while not rand_loc[0] == rand_loc[2] and rand_loc[1] == rand_loc[3]:
            rand_loc = np.random.choice(args['size'], 4)

    start_loc = rand_loc[:2]
    goal_loc = rand_loc[-2:]

    maze[tuple(start_loc)] = args['cell_type']['empty']
    maze[tuple(goal_loc)] = args['cell_type']['goal']

    return start_loc, goal_loc, maze
