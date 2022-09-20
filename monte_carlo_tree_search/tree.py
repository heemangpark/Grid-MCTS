import networkx as nx

from maze import *
from monte_carlo_tree_search.tree_functions import *


class Tree:
    def __init__(self, agent_id=0):
        self.g = nx.DiGraph()
        self.g.add_node(1, state=FROM[agent_id], N=1)
        self.maze = State()
        self.action_sequence = []
        _ = expand(1, self.g)

    # def grow(self):
    #     node_idx = 1
    #     while self.g.nodes[node_idx]['state'] != TO:
    #         prev_state = self.g.nodes[node_idx]['state']
    #         node_idx, action = select(node_idx, self.g)
    #         self.action_sequence.append(action)
    #         next_state = transition(prev_state, action)
    #         self.maze = State(next_state)
    #         # print("took action \"{}\" from {} & now at {}".format(action, prev_state, next_state))
    #         leaves = expand(node_idx, self.g)
    #         for leaf in leaves:
    #             sr = reward(random_rollout(self.g.nodes[leaf]['state'], 10))
    #             backup(leaf, self.g, sr)

    def grow(self):
        while True:
            idx, self.action_sequence = 1, []
            """selection"""
            while len(children(idx, self.g)) != 0:
                idx, a = select(idx, self.g)
                self.action_sequence.append(a)
            """terminal check on selected leaf"""
            if self.g.nodes[idx]['state'] == TO:
                break
            else:
                pass
            """expansion"""
            leaves = expand(idx, self.g)
            """backup"""
            for leaf in leaves:
                backup(leaf, self.g)

    def route(self):
        return self.action_sequence
