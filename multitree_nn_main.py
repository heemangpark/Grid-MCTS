import torch

from env.multi_env import maze_env
from mcts_nn.tree_multi import MultiTree
from rl.q_agent_loc import QAgent
from utils.arguments import maze_args
from utils.visualize import vis_route, vis_route_total


def runner(iterations, n_ag, maze_size, max_step):
    env = maze_env(maze_args, n_ag)
    env.size = maze_size

    agent = QAgent()
    agent.load_state_dict(torch.load('./sacred/grid_rand_binaryrwd.th', 'cuda'))

    for id in range(iterations):
        "tree search"
        _, _ = env.reset()
        tree = MultiTree(env, agent, n_ag)
        tree.grow(max_step)

        "visualize"
        total_seq = []
        for i in range(n_ag):
            individual_seq = [s[i] for s in tree.state_seq]
            total_seq.append(individual_seq)
            vis_route(env.maze[i], individual_seq, env.start_loc[i], env.goal_loc[i],
                      'agent_{}_{}th'.format(i, id + 1))
        vis_route_total(env.maze, total_seq, env.start_loc, env.goal_loc, 'total_{}th'.format(id + 1))


if __name__ == "__main__":
    runner(3, 2, 10, 100)
