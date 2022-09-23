import pickle

import networkx as nx
import numpy as np

from monte_carlo_tree_search.tree import Tree
from utils.arguments import get_fixed_maze_args
from utils.create_maze import create
from utils.vis_util import vis_route

if __name__ == "__main__":
    np.random.seed(42)
    args = get_fixed_maze_args()
    maze = create(get_fixed_maze_args())
    tree = Tree(args=args, maze=maze, agent_id=1)
    tree.grow()
    vis_route(args=args, maze=maze, seq=tree.state_sequence, agent_id=1)

    nx.write_gpickle(tree, './tree/tree.nx')
    pickle.dump(tree.action_sequence, open('./tree/route_{}.pkl'.format(len(tree.action_sequence)), 'wb'))
