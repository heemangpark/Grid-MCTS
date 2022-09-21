import dgl
import numpy as np

EMPTY = 0
OBSTACLE = 1
GOAL = 2
AG = 3

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
move = np.array([[0, 1], [0, -1], [-1, 0], [1, 0]])


class maze_env:
    def __init__(self, grid=10, obs_factor=.2):
        self.grid = grid
        self.obs_factor = obs_factor
        self.ag_loc = None
        self.maze = None
        self.base_graph = None
        self.time_limit = 100

    def reset(self):
        maze = np.zeros((self.grid, self.grid))
        obstacle = np.random.random((self.grid, self.grid)) < self.obs_factor
        maze[obstacle] = OBSTACLE

        goal_loc = np.random.choice(10, 2, replace=False)
        maze[tuple(goal_loc)] = GOAL

        rand_loc = np.random.choice(10, 4)
        if all(rand_loc[:2] == rand_loc[-2:]):
            while rand_loc[:2] != rand_loc[-2:]:
                rand_loc = np.random.choice(10, 4)

        ag_loc = rand_loc[:2]
        goal_loc = rand_loc[-2]

        maze[tuple(goal_loc)] = GOAL
        self.maze = maze
        self.base_graph = self.generate_base_graph(maze)

        self.ag_loc = ag_loc
        state = self.maze
        state[tuple(ag_loc)] = AG

        return state, self.get_avail_action(state)

    def mask(self, maze):
        ag_loc = maze.argmax()
        mask = []
        for a in [UP, DOWN, LEFT, RIGHT]:
            m = True
            if all(ag_loc + move[a] >= 0) and all(ag_loc + move[a] < self.grid):
                next_loc = ag_loc + move[a]
                if maze[tuple(next_loc)] == 0 or maze[tuple(next_loc)] == 2:
                    m = False

            mask.append(m)
        return mask

    def step(self, action):
        self.ag_loc = self.ag_loc + move[action]
        state = self.maze

        terminated = False
        reward = -1
        if state[tuple(self.ag_loc)] == GOAL:
            terminated = True
            reward = 10
        else:
            state[tuple(self.ag_loc)] = AG

        g = self.convert_maze_to_g(state, self.base_graph)

        return g, reward, self.get_avail_action(state), terminated

    def generate_base_graph(self, maze):
        g = dgl.DGLGraph()
        n_row = maze.shape[0]
        n_col = maze.shape[-1]
        g.add_nodes(n_row * n_col)

        vertical_from = np.arange(n_row * n_col - n_row)
        vertical_to = vertical_from + n_row
        g.add_edges(vertical_from, vertical_to)

        horizontal_from = np.arange(n_row * n_col)[:-1].reshape(-1)
        horizontal_to = horizontal_from + 1
        g.add_edges(horizontal_from, horizontal_to)
        return g

    def convert_maze_to_g(self, maze, g=None):
        if g is None:
            g = self.generate_base_graph(maze)

        g.ndata['feature'] = maze.reshape(-1, 1)

        return g
