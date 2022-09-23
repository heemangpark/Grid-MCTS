import argparse

import numpy as np


def get_fixed_maze_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--maze_x', default=10)
    parser.add_argument('--maze_y', default=10)
    parser.add_argument('--start', default=[[8, 1], [7, 8]])
    parser.add_argument('--goal', default=[1, 7])
    parser.add_argument('--obstacles', default=[[1, 1], [2, 1], [4, 4], [4, 5], [4, 6], [6, 5], [7, 5]])
    parser.add_argument('--difficulty', default=0.2)
    parser.add_argument('--cell_type', default={'empty': 0, 'obstacle': 1, 'start': 2, 'goal': 3})
    args = parser.parse_args()
    return args


def get_random_maze_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--maze_x', default=10)
    parser.add_argument('--maze_y', default=10)
    parser.add_argument('--difficulty', default=0.2)
    parser.add_argument('--cell_type', default={'empty': 0, 'obstacle': 1, 'goal': 2, 'agent': 3})
    parser.add_argument('--actions', default=np.array([[-1, 0], [1, 0], [0, -1], [0, 1]]))
    parser.add_argument('--action_type', default={'up': 0, 'down': 1, 'left': 2, 'right': 3})
    args = parser.parse_args()
    return args
