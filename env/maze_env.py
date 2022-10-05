from copy import deepcopy, copy

import dgl
import numpy as np
import torch

from env.maze_func import UP, DOWN, LEFT, RIGHT, move

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

        # if self.env_import:
        #     id = np.random.choice(list(set(range(1, 101)) - {"user_define"}))
        #     maze = np.load('../utils/sample_maps/maze_{}.npy'.format(id))
        #     self.maze = maze
        #     self.size = maze.shape[0]
        # TODO: AG loc? Goal loc: start loc?

        maze, ag_loc, start_loc, goal_loc = generate_maze(self.size, self.difficulty)
        if not check_feasibility(maze, ag_loc):
            while True:
                maze, ag_loc, start_loc, goal_loc = generate_maze(self.size, self.difficulty)
                if check_feasibility(maze, ag_loc):
                    break

        self.maze = maze
        self.ag_loc = ag_loc
        self.start_loc = start_loc
        self.goal_loc = goal_loc

        self.base_graph, state = self.generate_base_graph_loc(maze), copy(self.maze)
        ret_maze, ret_mask = self.convert_maze_to_g_loc(), get_mask(state, self.ag_loc)

        self.T = self.size * 4
        return ret_maze, ret_mask

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
        mask = get_mask(state, self.ag_loc)

        return g, reward, mask, terminated

    # def generate_base_graph(self, maze):
    #     g = dgl.DGLGraph()
    #     n_row = maze.shape[0]
    #     n_col = maze.shape[-1]
    #     g.add_nodes(n_row * n_col)
    #
    #     vertical_from = np.arange(n_row * n_col).reshape(n_row, -1)[:-1]
    #     vertical_from = vertical_from.reshape(-1)
    #     vertical_to = vertical_from + self.size
    #     g.add_edges(vertical_from, vertical_to)
    #     g.add_edges(vertical_to, vertical_from)
    #
    #     horizontal_from = np.arange(n_row * n_col).reshape(n_row, -1)[:, :-1]
    #     horizontal_from = horizontal_from.reshape(-1)
    #     horizontal_to = horizontal_from + 1
    #     g.add_edges(horizontal_from, horizontal_to)
    #     g.add_edges(horizontal_to, horizontal_from)
    #
    #     g.add_edges(range(n_row * n_col), range(n_row * n_col))
    #     return g

    def generate_base_graph_loc(self, maze):
        g = dgl.DGLGraph()

        obs_type = self.cell_type['obstacle']
        goal_type = self.cell_type['goal']
        obstacle_x, obstacle_y = (maze == obs_type).nonzero()
        goal_x, goal_y = (maze == goal_type).nonzero()

        n_obstacle = len(obstacle_x)
        g.add_nodes(n_obstacle + 1)

        obstacle_nf = np.stack([obstacle_x, obstacle_y], -1)  # coordination of obstacles
        goal_nf = np.stack([goal_x, goal_y], -1)  # coordination of goal

        init_nf = np.concatenate([obstacle_nf, goal_nf], 0) / self.size
        g.ndata['init_nf'] = torch.Tensor(init_nf)
        g.ndata['type'] = torch.Tensor([obs_type] * n_obstacle + [goal_type]).reshape(-1, 1)

        return g

    # def convert_maze_to_g(self, state):
    #     g = copy(self.base_graph)
    #     g.ndata['type'] = torch.Tensor(state.reshape(-1, 1))
    #     g.ndata['init_nf'] = torch.eye(4)[state.reshape(-1)]
    #     return g

    def convert_maze_to_g_loc(self):
        g = deepcopy(self.base_graph)
        g.add_nodes(1)
        n_nodes = g.number_of_nodes()

        obs_type = OBSTACLE
        goal_type = GOAL
        ag_type = AGENT

        g.ndata['init_nf'][-1] = torch.Tensor(self.ag_loc) / self.size
        g.ndata['type'] = torch.Tensor([obs_type] * (n_nodes - 2) + [goal_type] + [ag_type]).reshape(-1, 1)

        g.add_edges(range(n_nodes), n_nodes - 1)
        loc_gap = (g.ndata['init_nf'] - g.ndata['init_nf'][-1]).abs()
        g.edata['init_ef'] = loc_gap.sum(-1, keepdims=True)  # Manhattan distance
        g.edata['type'] = torch.Tensor([obs_type] * (n_nodes - 2) + [goal_type] + [ag_type]).reshape(-1, 1)

        return g


def generate_maze(size, difficulty):
    maze = np.zeros((size, size))
    obstacle = np.random.random((size, size)) < difficulty
    maze[obstacle] = OBSTACLE

    rand_loc = np.random.choice(size, 4)
    if rand_loc[0] == rand_loc[2] and rand_loc[1] == rand_loc[3]:
        while not rand_loc[0] == rand_loc[2] and rand_loc[1] == rand_loc[3]:
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
    for a in [UP, DOWN, LEFT, RIGHT]:  # 상 하 좌 우
        if (0 <= list(loc + move[a])[0] < size) and (
                0 <= list(loc + move[a])[1] < size):
            if maze[tuple(loc + move[a])] == EMPTY:
                m = False
            elif maze[tuple(loc + move[a])] == OBSTACLE:
                m = True
            elif maze[tuple(loc + move[a])] == GOAL:
                m = False
            else:
                m = None
        else:  # out of bound
            m = True
        mask.append(m)

    return torch.tensor(mask).reshape(1, -1)


def check_feasibility(maze, ag_loc):
    if get_mask(maze, ag_loc).sum() == 4:
        return False
    size = maze.shape[0]
    goal_loc = (maze == GOAL).nonzero()
    goal_loc = np.array(goal_loc).reshape(-1)
    visited = np.zeros(maze.shape, bool)

    locs = [ag_loc]
    feasible = False
    while len(locs) > 0:
        temp_loc = locs.pop()
        for a in ([0, 1], [0, -1], [1, 0], [-1, 0]):
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
    env.size = 20
    env.difficulty = 0.5
    env.reset()
    vis_map_only(env.maze, env.ag_loc, env.goal_loc)
    start_time = time()
    # feasible = check_feasibility(env.maze, env.ag_loc)
    # print(feasible)
    # print(time() - start_time)
