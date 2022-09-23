import networkx as nx

from env.maze_func import get_avail_action, terminated
from monte_carlo_tree_search.tree_functions import expand, children, select, backup


class Tree:
    def __init__(self, args, maze, agent_id):
        self.args = args
        self.maze = maze
        self.g = nx.DiGraph()
        self.g.add_node(1, state=args.start[agent_id - 1], N=1)
        self.state_sequence, self.action_sequence = [args.start[agent_id - 1]], []
        _ = expand(self.g, 1, get_avail_action(self.maze, args.start[agent_id - 1]))

    def grow(self):
        while True:
            idx = 1
            self.state_sequence = []
            self.action_sequence = []

            """selection"""
            while len(children(self.g, idx)) != 0:
                idx, a = select(self.g, idx)
                self.action_sequence.append(a)
                curr_state = self.g.nodes[idx]['state']
                self.state_sequence.append(curr_state)

            """terminal check on selected leaf"""
            if terminated(self.args, self.g.nodes[idx]['state']):
                break
            else:
                pass

            """expansion"""
            curr_state = self.g.nodes[idx]['state']
            leaves = expand(self.g, idx, avail_actions=get_avail_action(self.maze, curr_state))

            """backup"""
            for leaf in leaves:
                backup(self.args, self.g, leaf)
