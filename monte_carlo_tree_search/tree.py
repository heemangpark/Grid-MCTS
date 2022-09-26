import networkx as nx

from env.maze_func import get_avail_action, terminated
from monte_carlo_tree_search.tree_functions import expand, children, select, backup


class Tree:
    def __init__(self, start, goal, maze):
        self.goal = goal
        self.maze = maze
        self.g = nx.DiGraph()
        self.g.add_node(1, state=start, N=1)
        _ = expand(self.g, 1, get_avail_action(self.maze, start))
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
                backup(self.goal, self.g, leaf)
