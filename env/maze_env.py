from copy import deepcopy, copy

import dgl
import numpy as np
import torch

from utils.create_maze import create_randomly


class maze_env:
    def __init__(self, args, time_limit):
        self.t = 0
        self.time_limit = time_limit
        self.init_loc, self.ag_loc, self.goal_loc = None, None, None
        self.maze, self.base_graph = None, None
        self.args = args

    def reset(self):
        self.t = 0
        start_loc, self.goal_loc, self.maze = create_randomly(self.args)
        self.init_loc, self.ag_loc = start_loc, start_loc
        self.base_graph = self.generate_base_graph_loc(self.maze)

        state = copy(self.maze)
        state[tuple(self.ag_loc)] = self.args.cell_type['agent']

        mask = self.mask(state)
        if mask.sum() == 4:
            self.reset()

        return self.convert_maze_to_g_loc(state), self.mask(state)

    def mask(self, maze):
        ag_loc, mask = self.ag_loc, []
        for a in list(self.args.action_type.values()):
            m = True
            temp_next_loc = ag_loc + self.args.actions[a]
            if (0 <= temp_next_loc[0] < self.args.maze_x) and (0 <= temp_next_loc[1] < self.args.maze_y):
                next_loc = temp_next_loc
            else:
                next_loc = ag_loc
            if maze[tuple(next_loc)] == self.args.cell_type['empty'] or self.args.cell_type['goal']:
                m = False
            mask.append(m)
        return torch.tensor(mask).reshape(1, -1)

    def step(self, action):
        self.t += 1

        """transition"""
        temp_next_loc = self.ag_loc + self.args.actions[action]
        if (0 <= temp_next_loc[0] < self.args.maze_x) and (0 <= temp_next_loc[1] < self.args.maze_y):
            self.ag_loc = temp_next_loc
        else:
            pass
        # self.ag_loc = self.ag_loc + self.args.actions[action]

        state = copy(self.maze)
        terminated = False
        reward = -1
        if state[tuple(self.ag_loc)] == self.args.cell_type['goal']:
            terminated = True
            reward = 10
        else:
            if np.abs(self.goal_loc - self.ag_loc).sum() < 4:
                reward = 0
            else:
                pass

        state[tuple(self.ag_loc)] = self.args.cell_type['agent']
        g = self.convert_maze_to_g_loc(state)
        if self.t > self.time_limit:
            terminated = True

        mask = self.mask(state)

        return g, reward, mask, terminated

    def generate_base_graph(self, maze):
        g = dgl.DGLGraph()
        n_row = maze.shape[0]
        n_col = maze.shape[-1]
        g.add_nodes(n_row * n_col)

        vertical_from = np.arange(n_row * n_col).reshape(n_row, -1)[:-1]
        vertical_from = vertical_from.reshape(-1)
        vertical_to = vertical_from + self.args.maze_x
        g.add_edges(vertical_from, vertical_to)
        g.add_edges(vertical_to, vertical_from)

        horizontal_from = np.arange(n_row * n_col).reshape(n_row, -1)[:, :-1]
        horizontal_from = horizontal_from.reshape(-1)
        horizontal_to = horizontal_from + 1
        g.add_edges(horizontal_from, horizontal_to)
        g.add_edges(horizontal_to, horizontal_from)

        g.add_edges(range(n_row * n_col), range(n_row * n_col))
        return g

    def generate_base_graph_loc(self, maze):
        g = dgl.DGLGraph()

        obs_type = self.args.cell_type['obstacle']
        goal_type = self.args.cell_type['goal']
        obstacle_x, obstacle_y = (maze == obs_type).nonzero()
        goal_x, goal_y = (maze == goal_type).nonzero()

        n_obstacle = len(obstacle_x)
        g.add_nodes(n_obstacle + 1)

        obstacle_nf = np.stack([obstacle_x, obstacle_y], -1)
        goal_nf = np.stack([goal_x, goal_y], -1)

        init_nf = np.concatenate([obstacle_nf, goal_nf], 0) / self.args.maze_x
        g.ndata['init_nf'] = torch.Tensor(init_nf)
        g.ndata['type'] = torch.Tensor([obs_type] * n_obstacle + [goal_type]).reshape(-1, 1)
        return g

    def convert_maze_to_g(self, state):
        g = copy(self.base_graph)
        g.ndata['type'] = torch.Tensor(state.reshape(-1, 1))
        g.ndata['init_nf'] = torch.eye(4)[state.reshape(-1)]
        return g

    def convert_maze_to_g_loc(self, *args):
        g = deepcopy(self.base_graph)
        g.add_nodes(1)
        n_nodes = g.number_of_nodes()

        obs_type = self.args.cell_type['obstacle']
        goal_type = self.args.cell_type['goal']
        ag_type = self.args.cell_type['agent']

        g.ndata['init_nf'][-1] = torch.Tensor(self.ag_loc) / self.args.maze_x

        g.add_edges(range(n_nodes), n_nodes - 1)
        g.ndata['type'] = torch.Tensor([obs_type] * (n_nodes - 2) + [goal_type] + [ag_type]).reshape(-1, 1)
        g.edata['type'] = torch.Tensor([obs_type] * (n_nodes - 2) + [goal_type] + [ag_type]).reshape(-1, 1)

        return g
