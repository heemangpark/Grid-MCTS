import pickle

import networkx as nx
import numpy as np

from mcts.tree import Tree
from utils.arguments import maze_args
from utils.env_generator import create
from utils.visualize import vis_route

if __name__ == "__main__":
    start, goal, maze = create(maze_args)
    # start, goal, maze = [0, 0], [9, 9], np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                                               [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    #                                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                                               [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #                                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                                               [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    #                                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                                               [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #                                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                                               [1, 1, 1, 1, 1, 1, 1, 1, 1, 2]])
    tree = Tree(start, goal, maze)
    tree.grow()
    vis_route(maze_args, maze, tree.state_seq, start, goal, 'tree')

    nx.write_gpickle(tree, './tree/tree.nx')
    pickle.dump(tree.act_seq, open('./tree/route_{}.pkl'.format(len(tree.act_seq)), 'wb'))
