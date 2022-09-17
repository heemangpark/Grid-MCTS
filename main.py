from greedy_agent import *
from monte_carlo_tree_search.tree import *

if __name__ == "__main__":
    option = input("tree or greedy: ")
    if option == "tree":
        tree = Tree()
        tree.grow()
        route = tree.route()
        print(len(route))
    else:
        greedy_agent = Greedy_Agent()
        greedy_agent.play(rounds=100)
        print(greedy_agent.show_values())
