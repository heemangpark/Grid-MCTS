from config import *


def condition(coordinate):
    return False if (coordinate[0] < 0 or coordinate[0] == MAZE_ROWS) or (
            coordinate[1] < 0 or coordinate[1] == MAZE_COLS) else False if coordinate in OBSTACLES_LINE else True


class State:
    def __init__(self, state=FROM[0]):
        self.state = state
        self.terminal = False

    def reward(self):
        if self.state == TO:
            return 1
        elif self.state in OBSTACLES_LINE:
            return -1
        elif (self.state[0] < 0 or self.state[0] == MAZE_ROWS) or (self.state[1] < 0 or self.state[1] == MAZE_COLS):
            return -1
        else:
            return 0

    def is_terminal(self):
        self.terminal = True if self.state == TO else False

    def next_position(self, action):
        """action -> up down left right"""
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
        return self.state
