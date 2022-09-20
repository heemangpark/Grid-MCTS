import pickle
import time

from monte_carlo_tree_search.tree import *

if __name__ == "__main__":
    for trial in range(1, 11):
        start = int(time.time())
        tree = Tree()
        tree.grow()
        route = tree.route()
        nx.write_gpickle(tree, "tree_{}_{}".format(trial, int(time.time()) - start))
        with open('route_{}_{}.pkl'.format(trial, len(route)), 'wb') as f:
            pickle.dump(route, f)

    # """greedy"""
    # greedy_agent = Greedy_Agent()
    # # greedy_agent.play(rounds=100)
    # # print(greedy_agent.show_values())
