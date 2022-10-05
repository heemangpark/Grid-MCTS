import pickle

import networkx as nx

from env.maze_env import maze_env
from mcts.tree import Tree
from utils.arguments import maze_args
from utils.visualize import vis_route

if __name__ == "__main__":
    env = maze_env(maze_args)
    env.reset(size=20)
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
    tree = Tree(env)
    tree.grow()
    vis_route(env.maze, tree.state_seq, env.start_loc, env.goal_loc, 'tree')

    nx.write_gpickle(tree, './tree/tree.nx')
    pickle.dump(tree.act_seq, open('./tree/route_{}.pkl'.format(len(tree.act_seq)), 'wb'))
