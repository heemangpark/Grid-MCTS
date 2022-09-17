import numpy as np

from maze import *


class Greedy_Agent:

    def __init__(self):
        self.states = []
        self.actions = ["up", "down", "left", "right"]
        self.State = State()
        self.gamma = 0.2
        self.epsilon = 0.3
        self.state_values = {}
        for i in range(MAZE_ROWS):
            for j in range(MAZE_COLS):
                self.state_values[i, j] = 0

    def choose_action(self):
        """epsilon greedy"""
        action, max_reward = "", 0
        if np.random.uniform(0, 1) <= self.epsilon:
            action = np.random.choice(self.actions)
        else:
            for a in self.actions:
                nxt_reward = self.state_values[tuple(self.State.next_position(a))]
                if nxt_reward >= max_reward:
                    action = a
                    max_reward = nxt_reward
        return action

    def take_action(self, action):
        position = self.State.next_position(action)
        return State(state=position)

    def reset(self):
        self.states = []
        self.State = State()

    def play(self, rounds):
        i = 0
        while i < rounds:
            if self.State.terminal:
                reward = self.State.reward()
                self.state_values[tuple(self.State.state)] = reward
                print("Game End Reward", reward)
                for s in reversed(self.states):
                    reward = self.state_values[tuple(s)] + self.gamma * (reward - self.state_values[tuple(s)])
                    self.state_values[tuple(s)] = round(reward, 3)
                self.reset()
                i += 1
            else:
                action = self.choose_action()
                self.states.append(self.State.next_position(action))
                print("current coordination {} action {}".format(self.State.state, action))
                self.State = self.take_action(action)
                self.State.is_terminal()
                print("next state", self.State.state)
                print("------------------------------------------")

    def show_values(self):
        for i in range(0, MAZE_ROWS):
            print('-------------------------------------------------------------------------------------------')
            out = '| '
            for j in range(0, MAZE_COLS):
                out += str(self.state_values[i, j]).ljust(6) + ' | '
            print(out)
        print('-------------------------------------------------------------------------------------------')
