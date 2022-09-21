import pickle
import time

from monte_carlo_tree_search.tree import *
from config import maze, FROM
from utils.vis_util import vis_route

if __name__ == "__main__":
    for trial in range(1):
        start = int(time.time())
        tree = Tree(maze, init_locs=FROM)
        vis_route(tree)
        tree.grow()
        route = tree.route()

        vis_route(tree)
        nx.write_gpickle(tree, "tree_{}_{}.nx".format(trial, int(time.time()) - start))
        with open('route_{}_{}.pkl'.format(trial, len(route)), 'wb') as f:
            pickle.dump(route, f)

    # """greedy"""
    # greedy_agent = Greedy_Agent()
    # # greedy_agent.play(rounds=100)
    # # print(greedy_agent.show_values())
