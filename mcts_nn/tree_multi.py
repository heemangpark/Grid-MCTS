from copy import deepcopy
from itertools import combinations

import networkx as nx
import numpy as np

from env.maze_func import get_avail_action
from mcts_nn.tree_functions_multi import mask4tree, children, select, expand_joint, backup


class MultiTree:
    def __init__(self, env, agent, n_ag):
        self.env = env
        self.agent = agent
        self.args = env.args
        self.g = nx.DiGraph()
        self.g.add_node(1, state=self.env.start_loc, visited=0, Q=0, maze=deepcopy(env.maze))
        self.state_seq, self.act_seq = None, None
        self.n_ag = n_ag

    def grow(self, max_step=500):
        step = 1
        while step < max_step:
            idx = 1
            self.act_seq = []
            self.state_seq = []

            """selection"""
            while len(children(self.g, idx)) != 0:
                idx, a = select(self.g, idx, c=2)
                self.act_seq.append(a)
                curr_state = list(self.g.nodes[idx]['state'])
                self.state_seq.append(curr_state)

            """terminal check on selected leaf"""
            if all([self.env.goal_loc[k] == self.g.nodes[idx]['state'][k] for k in range(self.n_ag)]):
                print("after {} step reached goal".format(step))

            """expansion"""
            curr_state = self.g.nodes[idx]['state']
            curr_maze = self.g.nodes[idx]['maze']
            joint_avail_actions = [get_avail_action(curr_maze[i], curr_state[i]) for i in range(self.n_ag)]
            leaves = expand_joint(self.g, idx, joint_avail_actions)

            dead_end = any([len(joint_avail_actions[i]) == 0 for i in range(self.n_ag)])
            if dead_end:
                backup(self.g, idx, -1, 0)
            ags = [c for c in combinations(self.env.ag_loc, r=2)]
            collide_bool = [ags[k][0] == ags[k][1] for k in range(len(ags))]
            if any(collide_bool):
                backup(self.g, idx, -1, 0)

            """backup"""
            for leaf in leaves:
                self.env.ag_loc = self.g.nodes[leaf]['state']
                manhattan = [(abs(np.array(self.env.ag_loc) - np.array(self.env.goal_loc)))[i].sum()
                             for i in range(self.n_ag)]
                joint_masks = [mask4tree(m, l) for m, l in zip(self.env.maze, self.env.ag_loc)]
                leaf_r = 1 if np.array(manhattan).sum() == 0 else 0
                leaf_v = sum([self.agent.step(self.env.convert_maze_to_g_loc(i), joint_masks[i], tree_search=True).max()
                              for i in range(self.n_ag)]) / self.n_ag
                backup(self.g, leaf, leaf_r, leaf_v)

            step += 1
