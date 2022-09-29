import time
from math import inf

import networkx as nx

from env.maze_func import get_avail_action
from mcts_nn.tree_functions import expand, children, select, backup
from utils.visualize import vis_route


def is_inf(val): return True if val != -inf else False


class Tree:
    def __init__(self, args, env, agent, base_graph):
        self.idx = 1
        self.start = env.start_loc
        self.goal = env.goal_loc
        self.maze = env.maze
        self.g = nx.DiGraph()
        self.g.add_node(1, state=self.start, N=1)
        # _ = expand(self.g, 1, get_avail_action(self.maze, self.start))
        self.state_seq, self.act_seq = None, None
        self.base_graph = base_graph
        self.agent = agent
        self.args = args

    def grow(self):
        start_time = int(time.time())
        while True:
            self.idx = 1
            act_seq = []
            state_seq = []

            """selection"""
            while len(children(self.g, self.idx)) != 0:
                self.idx, a = select(self.g, self.idx)
                act_seq.append(a)
                curr_state = list(self.g.nodes[self.idx]['state'])
                state_seq.append(curr_state)

            self.state_seq = state_seq
            self.act_seq = act_seq

            """terminal check on selected leaf"""
            if (self.goal[0] == self.g.nodes[self.idx]['state'][0]) and (
                    self.goal[1] == self.g.nodes[self.idx]['state'][1]):
                break
                # print("GOAL REACHED -> MORE BACKUP")
            else:
                pass

            """expansion"""
            curr_state = self.g.nodes[self.idx]['state']
            leaves = expand(self.g, self.idx, avail_actions=get_avail_action(self.maze, curr_state))

            """backup"""
            for leaf in leaves:
                backup(self.agent, self.maze, self.base_graph, self.g, leaf)

            """visualize per step"""
            if int(time.time()) - start_time >= 10:
                vis_route(self.args, self.maze, self.state_seq, self.start, self.goal,
                          str(int(time.time()) - start_time))
