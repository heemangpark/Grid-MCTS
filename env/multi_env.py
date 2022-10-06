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
    def __init__(self, args, num_agent=3):
        self.args = args
        self.num_agent = num_agent
        self.difficulty = None
        self.size = None
        self.t = 0
        self.T = None
        self.ag_loc, self.start_loc, self.goal_loc = None, None, None
        self.maze, self.base_graph = None, None

        for key in args:
            setattr(self, key, args[key])

    def reset(self, size=None):
        if size is not None:
            self.size = size

        self.t = 0
        self.T = self.size * 4

        maze_zip, ag_loc, start_loc, goal_loc = generate_maze(self.size, self.difficulty, self.num_agent)
        _cf = [check_feasibility(maze_zip[idx], ag_loc[idx]) for idx in range(self.num_agent)]
        while not all(_cf):
            maze_zip, ag_loc, start_loc, goal_loc = generate_maze(self.size, self.difficulty, self.num_agent)
            _cf = [check_feasibility(maze_zip[idx], ag_loc[idx]) for idx in range(self.num_agent)]

        self.ag_loc = ag_loc
        self.start_loc = start_loc
        self.goal_loc = goal_loc
        self.maze = maze_zip

        self.base_graph = [self.generate_base_graph_loc(maze) for maze in self.maze]
        ret_g = [self.convert_maze_to_g_loc(i) for i in range(self.num_agent)]

        state = copy(self.maze)
        ret_mask = [get_mask(state[idx], self.ag_loc[idx]) for idx in range(self.num_agent)]

        return ret_g, ret_mask

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

    # def multi_step(self, actions):  # action should be joint action form
    #     self.t += 1
    #     moves = np.array([move[a] for a in actions])
    #     self.ag_loc = np.array(self.ag_loc) + moves
    #     state = copy(self.maze)
    #     terminated = [False for _ in range(self.num_agent)]
    #     reward = -1
    #
    #     grid_type = [state[tuple(agl)] for agl in self.ag_loc]
    #     for g in range(self.num_agent):
    #         if grid_type[g] == self.cell_type['goal']:
    #             terminated[g] = True
    #     if all(terminated):
    #         reward = 10
    #     else:
    #         nearby_bool = [None for _ in range(self.num_agent)]
    #         nearby_goal = [np.abs(self.goal_loc - agl).sum() for agl in self.ag_loc]
    #
    #         if np.abs(self.goal_loc - self.ag_loc).sum() < 4:
    #             reward = 0
    #         else:
    #             pass
    #
    #     g = self.convert_maze_to_g_loc()
    #
    #     if self.t > self.T:
    #         terminated = True
    #     mask = get_mask(state, self.ag_loc)
    #
    #     return g, reward, mask, terminated

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

    def convert_maze_to_g_loc(self, i):
        g = deepcopy(self.base_graph[i])
        g.add_nodes(1)
        n_nodes = g.number_of_nodes()

        obs_type = OBSTACLE
        goal_type = GOAL
        ag_type = AGENT

        g.ndata['init_nf'][-1] = torch.Tensor(self.ag_loc[i]) / self.size
        g.ndata['type'] = torch.Tensor([obs_type] * (n_nodes - 2) + [goal_type] + [ag_type]).reshape(-1, 1)

        g.add_edges(range(n_nodes), n_nodes - 1)
        loc_gap = (g.ndata['init_nf'] - g.ndata['init_nf'][-1]).abs()
        g.edata['init_ef'] = loc_gap.sum(-1, keepdims=True)  # Manhattan distance
        g.edata['type'] = torch.Tensor([obs_type] * (n_nodes - 2) + [goal_type] + [ag_type]).reshape(-1, 1)

        return g


def generate_maze(size, difficulty, num_agent):
    maze = np.zeros((size, size))
    obstacle = np.random.random((size, size)) < difficulty
    maze[obstacle] = OBSTACLE
    maze_zip = [copy(maze) for _ in range(num_agent)]  # 공유되는 벽 정보만 포함하는 미로 여러개 생성

    s_g_idx = np.random.choice(list(range(size ** 2)), num_agent * 2, replace=False)
    entire_loc = [[i, j] for i in range(size) for j in range(size)]

    ag_loc = [[] for _ in range(num_agent)]
    goal_loc = [[] for _ in range(num_agent)]

    for a in range(num_agent):
        ag_loc[a] = entire_loc[s_g_idx[a]]
    start_loc = ag_loc
    for g in range(num_agent):
        goal_loc[g] = entire_loc[s_g_idx[g - num_agent]]

    for a in range(num_agent):
        for maze in maze_zip:
            maze[tuple(ag_loc[a])] = EMPTY
    for g in range(num_agent):
        maze_zip[g][tuple(goal_loc[g])] = GOAL

    return maze_zip, ag_loc, start_loc, goal_loc


def get_mask(maze, loc):
    size = maze.shape[0]
    mask = []
    for a in [UP, DOWN, LEFT, RIGHT]:
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
