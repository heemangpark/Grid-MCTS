import torch

from env.multi_env import maze_env
from mcts_nn.tree_multi import MultiTree
from rl.q_agent_loc import QAgent
from utils.arguments import maze_args
from utils.visualize import vis_route, vis_route_total

if __name__ == "__main__":
    for id in range(1):
        n_ag = 1
        env = maze_env(maze_args, n_ag)
        env.size = 20
        env.difficulty = .5
        max_step = 1000

        agent = QAgent()
        agent.load_state_dict(torch.load('./sacred/grid_rand_binaryrwd.th', 'cuda'))

        g, mask = env.reset()
        tree = MultiTree(env, agent, n_ag)
        tree.grow(max_step=max_step)

        seq = []
        for i in range(n_ag):
            individual_seq = [s[i] for s in tree.state_seq]
            seq.append(individual_seq)
            vis_route(env.maze[i], individual_seq, env.start_loc[i], env.goal_loc[i],
                      'multi_tree_{}_{}'.format(i, id + 1))

        vis_route_total(env.maze, seq, env.start_loc, env.goal_loc, 'multi_tree_total')
