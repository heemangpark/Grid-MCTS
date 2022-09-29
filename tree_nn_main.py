import time

import dgl
import networkx as nx
import numpy as np
import torch

from env.maze_env import maze_env
from mcts_nn.tree import Tree
from rl.q_agent_loc import QAgent
from utils.arguments import maze_args
from utils.visualize import vis_route


def generate_base_graph_loc(maze):
    g = dgl.DGLGraph()
    obstacle_x, obstacle_y = (maze == 1).nonzero()
    goal_x, goal_y = (maze == 2).nonzero()

    n_obstacle = len(obstacle_x)
    g.add_nodes(n_obstacle + 1)

    obstacle_nf = np.stack([obstacle_x, obstacle_y], -1)  # coordination of obstacles
    goal_nf = np.stack([goal_x, goal_y], -1)  # coordination of goal

    init_nf = np.concatenate([obstacle_nf, goal_nf], 0) / 20
    g.ndata['init_nf'] = torch.Tensor(init_nf)
    g.ndata['type'] = torch.Tensor([1] * n_obstacle + [2]).reshape(-1, 1)

    return g


if __name__ == "__main__":
    args = maze_args
    args['size'] = 20
    env = maze_env(args, T=4 * args['size'])

    agent = QAgent(in_dim=2, embedding_dim=64)
    agent.to(agent.device)
    agent.load_state_dict(torch.load('./saved/grid_5_99000.th'))

    g, mask = env.reset()
    maze_to_graph = generate_base_graph_loc(env.maze)
    tree = Tree(args, env, agent, maze_to_graph)
    start_time = int(time.time())
    tree.grow()

    """all done"""
    vis_route(maze_args, env.maze, tree.state_seq, env.start_loc, env.goal_loc, 'tree')
    nx.write_gpickle(tree, './tree/tree.nx')
    np.save('./tree/route', np.array(tree.act_seq))

# route = np.load('./tree/route.npy')
# print(route)