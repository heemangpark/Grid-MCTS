import networkx as nx
import numpy as np

from env.maze_func import get_avail_action
from mcts_nn.tree_functions import mask4tree, children, select
from mcts_nn.tree_functions_multi import expand_joint, backup


class MultiTree:
    def __init__(self, env, agent, n_ag):
        self.env = env
        self.agent = agent
        self.args = env.args
        self.g = nx.DiGraph()
        self.g.add_node(1, state=self.env.start_loc, visited=1, Q=0)
        self.state_seq, self.act_seq = None, None
        self.n_ag = n_ag

    def grow(self, max_step=500):
        step = 1
        while True:
            idx = 1
            act_seq = []
            state_seq = []

            """selection"""
            while len(children(self.g, idx)) != 0:
                idx, a = select(self.g, idx, c=10)
                act_seq.append(a)
                curr_state = list(self.g.nodes[idx]['state'])
                state_seq.append(curr_state)

            self.state_seq = state_seq
            self.act_seq = act_seq

            """terminal check on selected leaf"""
            if self.env.goal_loc == self.g.nodes[idx]['state']:
                print()
                break
            else:
                pass

            """expansion"""
            curr_state = self.g.nodes[idx]['state']
            joint_avail_actions = [get_avail_action(self.env.maze[i], curr_state[i]) for i in range(self.n_ag)]
            leaves = expand_joint(self.g, idx, joint_avail_actions)

            """backup"""
            for leaf in leaves:
                self.env.ag_loc = self.g.nodes[leaf]['state']
                dist_to_goal = abs(np.array(self.env.ag_loc) - np.array(self.env.goal_loc)).sum()
                temp_maze = np.zeros((self.env.size, self.env.size))
                penalty = 0
                for loc in self.env.ag_loc:
                    if temp_maze[tuple(loc)] == 1:
                        penalty = -10
                        break
                    temp_maze[tuple(loc)] = 1

                if dist_to_goal == 0:
                    leaf_r = 10
                elif dist_to_goal == 1:
                    leaf_r = 0
                else:
                    leaf_r = -1

                leaf_r += penalty

                joint_masks = [mask4tree(m, l) for m, l in zip(self.env.maze, self.env.ag_loc)]
                leaf_maxq = sum(
                    [self.agent.step(self.env.convert_maze_to_g_loc(i), joint_masks[i], tree_search=True).max() for
                     i in range(self.n_ag)]) / self.n_ag
                backup(self.g, leaf, leaf_r, leaf_maxq)

            step += 1
            if step >= max_step:
                break

            # """visualize per step"""
            # if step % 50 == 0:
            #     vis_route(self.env.maze, self.state_seq, self.env.start_loc, self.env.goal_loc, step)
