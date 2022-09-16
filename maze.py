import numpy as np

"""Global variables"""
MAZE_ROWS = 10
MAZE_COLS = 10
START = [8, 1]
WIN_STATE = [1, 7]
DETERMINISTIC = True
OBSTACLES = [[[1, 1], [2, 1]],
             [[4, 4], [4, 5], [4, 6]],
             [[6, 5], [7, 5]]]
OBSTACLES_LINE = [1, 1], [2, 1], [4, 4], [4, 5], [4, 6], [6, 5], [7, 5]


def condition(coordinate):
    """check out-of-bound first"""
    if (coordinate[0] < 0 or coordinate[0] == MAZE_ROWS) or (coordinate[1] < 0 or coordinate[1] == MAZE_COLS):
        return False  # coordinate is out of bound
    else:
        if coordinate in OBSTACLES_LINE:
            return False  # agents are bumped against a wall
        else:
            return True


class Maze:
    def __init__(self, state=START):
        self.maze = np.zeros([MAZE_ROWS, MAZE_COLS])
        self.state = state
        self.terminal = False
        self.determine = DETERMINISTIC

    def reward(self):
        return 1 if self.state == WIN_STATE else 0

    def next_position(self, action):
        """action -> up down left right"""
        if self.determine:
            if action == "up":
                next_state = (self.state[0] - 1, self.state[1])
                if not condition(self.state):
                    next_state = self.state
            elif action == "down":
                next_state = (self.state[0] + 1, self.state[1])
                if not condition(self.state):
                    next_state = self.state
            elif action == "left":
                next_state = (self.state[0], self.state[1] - 1)
                if not condition(self.state):
                    next_state = self.state
            else:
                next_state = (self.state[0], self.state[1] + 1)
                if not condition(self.state):
                    next_state = self.state
        else:
            raise NotImplementedError
        return next_state

    def visualize(self):
        self.maze[self.state] = 1
        for i in range(0, MAZE_ROWS):
            print('-----------------')
            out = '| '
            for j in range(0, MAZE_COLS):
                if self.maze[i, j] == 1:
                    token = '*'
                if self.maze[i, j] == -1:
                    token = 'z'
                if self.maze[i, j] == 0:
                    token = '0'
                out += token + ' | '
            print(out)
        print('-----------------')
