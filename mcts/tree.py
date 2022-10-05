import networkx as nx
import numpy as np

from env.maze_func import get_avail_action, terminated
from mcts.tree_functions import expand, children, select, backup


class Tree:
    def __init__(self, env):
        self.goal = env.goal_loc
        self.maze = env.maze
        self.g = nx.DiGraph()
        self.g.add_node(1, state=env.start_loc, N=1)
        self.state_seq, self.act_seq = None, None

    def grow(self):
        while True:
            idx = 1
            state_seq = []
            act_seq = []

            """selection"""
            while len(children(self.g, idx)) != 0:
                idx, a = select(self.g, idx)
                act_seq.append(a)
                curr_state = list(self.g.nodes[idx]['state'])
                state_seq.append(curr_state)

            self.state_seq = state_seq
            self.act_seq = act_seq

            """terminal check on selected leaf"""
            if terminated(self.goal, self.g.nodes[idx]['state']):
                break
            else:
                pass

            """expansion"""
            curr_state = self.g.nodes[idx]['state']
            leaves = expand(self.g, idx, avail_actions=get_avail_action(self.maze, curr_state))

            """backup"""
            for leaf in leaves:
                leaf_r = 1000 if self.maze[tuple(self.g.nodes[leaf]['state'])] == 2 else distance_score(
                    self.g.nodes[leaf]['state'], self.goal)
                backup(self.goal, self.g, leaf, leaf_r)


def distance_score(loc1, loc2):
    dist = -np.sqrt((loc1[0] - loc2[0]) ** 2 + (loc1[1] - loc2[1]) ** 2)
    return 10 if dist > -1 else dist
