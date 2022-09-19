from greedy_agent import *
from monte_carlo_tree_search.tree import *

if __name__ == "__main__":
    tree = Tree()
    tree.grow()
    route = tree.route()
    print(route)
    print(len(route))

    """greedy"""
    greedy_agent = Greedy_Agent()
    # greedy_agent.play(rounds=100)
    # print(greedy_agent.show_values())
