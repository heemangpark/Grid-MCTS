import pickle

import networkx as nx

from mcts.tree import Tree
from utils.arguments import maze_args
from utils.env_generator import create
from utils.visualize import vis_route

if __name__ == "__main__":
    start, goal, maze = create(maze_args)
    tree = Tree(start, goal, maze)
    tree.grow()
    vis_route(maze_args, maze, tree.state_seq, start, goal, 'tree')

    nx.write_gpickle(tree, './tree/tree.nx')
    pickle.dump(tree.act_seq, open('./tree/route_{}.pkl'.format(len(tree.act_seq)), 'wb'))
