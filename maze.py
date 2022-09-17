import numpy as np

from config import *


def condition(coordinate):
    """check out-of-bound first"""
    if (coordinate[0] < 0 or coordinate[0] == MAZE_ROWS) or (coordinate[1] < 0 or coordinate[1] == MAZE_COLS):
        return False
    else:
        return False if coordinate in OBSTACLES_LINE else True


class State:
    def __init__(self, state=FROM):
        self.maze = np.zeros([MAZE_ROWS, MAZE_COLS])
        self.state = state
        self.terminal = False
        self.determine = DETERMINISTIC

    def reward(self):
        if self.state == TO:
            return 1
        elif self.state in OBSTACLES_LINE:
            return -0.5
        elif (self.state[0] < 0 or self.state[0] == MAZE_ROWS) or (self.state[1] < 0 or self.state[1] == MAZE_COLS):
            return -0.5
        else:
            return 0

    def is_terminal(self):
        self.terminal = True if self.state == TO else False

    def next_position(self, action):
        """action -> up down left right"""
        if self.determine:
            if action == "up":
                temp_state = [self.state[0] - 1, self.state[1]]
                if condition(temp_state):
                    self.state = temp_state
            elif action == "down":
                temp_state = [self.state[0] + 1, self.state[1]]
                if condition(temp_state):
                    self.state = temp_state
            elif action == "left":
                temp_state = [self.state[0], self.state[1] - 1]
                if condition(temp_state):
                    self.state = temp_state
            else:
                temp_state = [self.state[0], self.state[1] + 1]
                if condition(temp_state):
                    self.state = temp_state
        else:
            raise NotImplementedError
        return self.state

    # def visualize(self):
    #     self.maze[self.state] = 1
    #     for i in range(0, MAZE_ROWS):
    #         print('-----------------')
    #         out = '| '
    #         for j in range(0, MAZE_COLS):
    #             if self.maze[i, j] == 1:
    #                 token = '*'
    #             if self.maze[i, j] == -1:
    #                 token = 'z'
    #             if self.maze[i, j] == 0:
    #                 token = '0'
    #             out += token + ' | '
    #         print(out)
    #     print('-----------------')
