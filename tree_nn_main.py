import networkx as nx
import numpy as np
import torch

from env.maze_env import maze_env
from mcts_nn.tree import Tree
from rl.q_agent_loc import QAgent
from utils.arguments import maze_args
from utils.visualize import vis_route


def runner(iterations, maze_size, max_step):
    env = maze_env(maze_args)
    env.size = maze_size
    max_step = max_step
    for id in range(iterations):
        agent = QAgent()
        agent.load_state_dict(torch.load('./sacred/rand_best.th', 'cuda'))

        g, mask = env.reset()
        tree = Tree(env, agent)
        tree.grow(max_step=max_step)

        vis_route(env.maze, tree.state_seq, env.start_loc, env.goal_loc, 'tree_{}'.format(id + 1))
        nx.write_gpickle(tree, './tree/tree_{}.nx'.format(id + 1))
        np.save('./tree/route_{}'.format(id + 1), np.array(tree.act_seq))


def loader(load_id):
    return nx.read_gpickle('./tree/tree_{}.nx'.format(load_id))


if __name__ == "__main__":
    runner(iterations=5, maze_size=50, max_step=10000)
    # loaded_model = loader(1)
    # loaded_tree = loaded_model.g
    print("")
