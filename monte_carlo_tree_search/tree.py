import networkx as nx

from maze import *
from monte_carlo_tree_search.tree_functions import *


class Tree:
    def __init__(self, agent_id=0):
        self.g = nx.DiGraph()
        self.g.add_node(1, state=FROM[agent_id], N=0)
        self.maze = State()
        self.action_sequence = []
        self.state_sequence = []

    def grow(self):
        node_idx = 1
        _ = expand(node_idx, self.g)
        self.state_sequence.append(self.g.nodes[1]['state'])

        while self.g.nodes[node_idx]['state'] != TO:
            prev_state = self.g.nodes[node_idx]['state']
            node_idx, action = select(node_idx, self.g)
            self.action_sequence.append(action)
            next_state = transition(prev_state, action)
            self.state_sequence.append(next_state)

            self.maze = State(next_state)
            # print("took action \"{}\" from {} & now at {}".format(action, prev_state, next_state))
            leaves = expand(node_idx, self.g)
            for leaf in leaves:
                sr = reward(random_rollout(self.g.nodes[leaf]['state'], 10))
                backup(leaf, self.g, sr)

    def route(self):
        return self.action_sequence
