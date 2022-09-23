import numpy as np


def create(args):
    maze = np.zeros((args.maze_x, args.maze_y))
    for s in args.start:
        maze[tuple(s)] = args.cell_type['start']
    maze[tuple(args.goal)] = args.cell_type['goal']
    for o in args.obstacles:
        maze[tuple(o)] = args.cell_type['obstacle']
    return maze


def create_randomly(args):
    maze = np.zeros((args.maze_x, args.maze_y))
    obstacle = np.random.random((args.maze_x, args.maze_y)) < args.difficulty
    maze[obstacle] = args.cell_type['obstacle']
    rand_loc = np.random.choice(args.maze_x, 4)
    if rand_loc[0] == rand_loc[2] and rand_loc[1] == rand_loc[3]:
        while not rand_loc[0] == rand_loc[2] and rand_loc[1] == rand_loc[3]:
            rand_loc = np.random.choice(10, 4)
    start_loc, goal_loc = rand_loc[:2], rand_loc[-2:]
    maze[tuple(start_loc)] = args.cell_type['empty']
    maze[tuple(goal_loc)] = args.cell_type['goal']
    return start_loc, goal_loc, maze
