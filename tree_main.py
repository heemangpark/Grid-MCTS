import pickle

import networkx as nx

from monte_carlo_tree_search.tree import Tree
from utils.arguments import maze_args
from utils.sample_map import grid
from utils.visualize import vis_route

if __name__ == "__main__":
    start, goal, maze = grid(maze_args)
    tree = Tree(start, goal, maze)
    tree.grow()
    vis_route(maze_args, maze, tree.state_seq, start, goal)

    nx.write_gpickle(tree, './tree/tree.nx')
    pickle.dump(tree.act_seq, open('./tree/route_{}.pkl'.format(len(tree.act_seq)), 'wb'))
