import torch

from env.maze_env import maze_env
from rl.q_agent_loc import QAgent
from utils.arguments import maze_args
from utils.visualize import vis_route


def evaluate(size, vis):
    args = maze_args
    env = maze_env(args)
    env.size = size

    for eval_id in range(vis):
        g, mask = env.reset()
        seq = []
        while True:
            action = agent.step(g, mask, greedy=True)
            ng, r, n_mask, t = env.step(action)
            seq.append(env.ag_loc)
            g, mask = ng, n_mask
            if t:
                break
        vis_route(args, env.maze, seq, env.start_loc, env.goal_loc, eval_id + 1)


if __name__ == "__main__":
    agent = QAgent(in_dim=2, embedding_dim=64)
    agent.load_state_dict(torch.load('./saved/grid_random_73000.th', 'cuda'))
    evaluate(size=10, vis=10)
