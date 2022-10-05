import networkx as nx
import numpy as np
import torch

from env.multi_env import maze_env
from mcts_nn.tree_multi import MultiTree
from rl.q_agent_loc import QAgent
from utils.arguments import maze_args
from utils.visualize import vis_route

if __name__ == "__main__":
    for id in range(1):
        n_ag = 2
        env = maze_env(maze_args, n_ag)
        env.size = 10
        max_step = 100

        agent = QAgent()
        agent.load_state_dict(torch.load('./sacred/rand_best.th', 'cuda'))

        g, mask = env.reset()
        tree = MultiTree(env, agent, n_ag)
        tree.grow(max_step=max_step)
        for i in range(n_ag):
            individual_seq = [s[i] for s in tree.state_seq]
            vis_route(env.maze[i], individual_seq, env.start_loc[i], env.goal_loc[i], 'tree_ag{}_{}'.format(i, id + 1))

    # nx.write_gpickle(tree, './tree/tree.nx')
    # np.save('./tree/route', np.array(tree.act_seq))
