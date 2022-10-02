import networkx as nx

from env.maze_func import get_avail_action
from mcts_nn.tree_functions import expand, children, select, backup
from utils.visualize import vis_route


class Tree:
    def __init__(self, args, env, agent, base_graph):
        self.idx = 1
        self.start = env.start_loc
        self.goal = env.goal_loc
        self.maze = env.maze
        self.g = nx.DiGraph()
        self.g.add_node(1, state=self.start, N=1)
        self.state_seq, self.act_seq = None, None
        self.base_graph = base_graph
        self.agent = agent
        self.args = args

    def grow(self):
        step = 1
        while True:
            idx = 1
            act_seq = []
            state_seq = []

            """selection"""
            while len(children(self.g, idx)) != 0:
                idx, a = select(self.g, idx)
                act_seq.append(a)
                curr_state = list(self.g.nodes[idx]['state'])
                state_seq.append(curr_state)

            # self.maze[tuple(self.g.nodes[idx]['state'])] = 1
            self.state_seq = state_seq
            self.act_seq = act_seq

            """terminal check on selected leaf"""
            if all(self.goal == self.g.nodes[idx]['state']):
                break
            else:
                pass

            """expansion"""
            curr_state = self.g.nodes[idx]['state']
            leaves = expand(self.g, idx, avail_actions=get_avail_action(self.maze, curr_state))

            """backup"""
            for leaf in leaves:
                backup(self.agent, self.base_graph, self.g, leaf)

            step += 1
            """visualize per step"""
            if step % 50 == 0:
                vis_route(self.args, self.maze, self.state_seq, self.start, self.goal, step)
                print(idx)
