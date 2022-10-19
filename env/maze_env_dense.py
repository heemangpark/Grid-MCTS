from copy import deepcopy, copy

import dgl
import numpy as np
import torch

from env.maze_func import UP, DOWN, LEFT, RIGHT, move, generate_dense_graph

EMPTY = 0
OBSTACLE = 1
GOAL = 2
AGENT = 3


class maze_env:
    def __init__(self, args):
        self.args = args
        self.t = 0
        self.T = None
        self.size = None
        self.difficulty = None
        self.ag_loc, self.start_loc, self.goal_loc = None, None, None
        self.maze, self.base_graph = None, None
        self.env_import = False

        for key in args:
            setattr(self, key, args[key])

    def reset(self, size=None):
        if size is not None:
            self.size = size
        self.t = 0
        # TODO: AG loc? Goal loc: start loc?

        maze, ag_loc, start_loc, goal_loc = generate_maze(self.size, self.difficulty)
        if not check_feasibility(maze, ag_loc):
            while True:
                maze, ag_loc, start_loc, goal_loc = generate_maze(self.size, self.difficulty)
                if check_feasibility(maze, ag_loc):
                    break

        self.base_graph = generate_dense_graph(maze)
        self.maze = maze
        self.ag_loc = ag_loc
        self.start_loc = start_loc
        self.goal_loc = goal_loc

        self.T = self.size * 4

        temp_maze = deepcopy(maze)
        temp_maze[tuple(ag_loc)] = AGENT
        ret_state = generate_dense_graph(temp_maze)
        return ret_state, None

    def step(self, action):
        self.t += 1
        self.ag_loc = self.ag_loc + move[action]
        state = copy(self.maze)
        terminated = False
        reward = -1

        if state[tuple(self.ag_loc)] == self.cell_type['goal']:
            terminated = True
            reward = 10
        else:
            if np.abs(self.goal_loc - self.ag_loc).sum() < 4:
                reward = 0
            else:
                pass

        g = self.convert_maze_to_g_loc()

        if self.t > self.T:
            terminated = True
        mask = None

        return g, reward, mask, terminated



def generate_maze(size, difficulty):
    maze = np.zeros((size, size))
    obstacle = np.random.random((size, size)) < difficulty
    maze[obstacle] = OBSTACLE

    rand_loc = np.random.choice(size, 4)
    if all(rand_loc[:2] == rand_loc[-2:]):
        while not all(rand_loc[:2] == rand_loc[-2:]):
            rand_loc = np.random.choice(size, 4)

    ag_loc = rand_loc[:2]
    start_loc = rand_loc[:2]
    goal_loc = rand_loc[-2:]
    maze[tuple(ag_loc)] = EMPTY
    maze[tuple(goal_loc)] = GOAL
    return maze, ag_loc, start_loc, goal_loc


def get_mask(maze, loc):
    size = maze.shape[0]
    mask = []
    for a in [[0, 1], [0, -1], [1, 0], [-1, 0], [1, 1], [1, -1], [-1, 1], [-1, -1]]:
        a = np.array(a)
        if (0 <= list(loc + a)[0] < size) and (
                0 <= list(loc + a)[1] < size):
            if maze[tuple(loc + a)] == EMPTY:
                m = False
            elif maze[tuple(loc + a)] == OBSTACLE:
                m = True
            elif maze[tuple(loc + a)] == GOAL:
                m = False
            else:
                m = None
        else:  # out of bound
            m = True
        mask.append(m)

    return torch.tensor(mask).reshape(1, -1)


def check_feasibility(maze, ag_loc):
    if get_mask(maze, ag_loc).sum() == 8:
        return False
    size = maze.shape[0]
    goal_loc = (maze == GOAL).nonzero()
    goal_loc = np.array(goal_loc).reshape(-1)
    visited = np.zeros(maze.shape, bool)

    locs = [ag_loc]
    feasible = False
    while len(locs) > 0:
        temp_loc = locs.pop()
        for a in [[0, 1], [0, -1], [1, 0], [-1, 0], [1, 1], [1, -1], [-1, 1], [-1, -1]]:
            new_loc = temp_loc + np.array(a)
            if all(new_loc >= 0) and all(new_loc < size):
                if maze[tuple(new_loc)] == OBSTACLE:
                    continue

                if not visited[tuple(new_loc)]:
                    visited[tuple(new_loc)] = True
                    locs.append(new_loc)

                if all(new_loc == goal_loc):
                    feasible = True
                    break

    return feasible


if __name__ == '__main__':
    from utils.arguments import maze_args
    from utils.visualize import vis_map_only
    from time import time

    EMPTY = 0
    OBSTACLE = 1
    GOAL = 2
    AGENT = 3

    env = maze_env(maze_args)
    env.size = 10
    env.difficulty = 0.5
    env.reset()
    vis_map_only(env.maze, env.ag_loc, env.goal_loc)
    start_time = time()
    # feasible = check_feasibility(env.maze, env.ag_loc)
    # print(feasible)
    # print(time() - start_time)
