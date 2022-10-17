import time

from env.maze_env import maze_env
from prev_codes.mcts.tree import Tree
from utils.arguments import maze_args
from utils.visualize import vis_route

if __name__ == "__main__":
    s = time.time()
    env = maze_env(maze_args)
    env.size = 30
    env.reset()
    tree = Tree(env, max_step=10000)
    tree.grow()
    e = time.time()
    print(e - s)
    vis_route(env.maze, tree.state_seq, env.start_loc, env.goal_loc, 'tree_only')
