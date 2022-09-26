import torch

from env.maze_env import maze_env
from rl.q_agent_loc import QAgent
from utils.arguments import maze_args
from utils.visualize import vis_route

"""
===========================                                   
==  train_arguments      ==
==  difficulty: 0.3      ==
==  size: 5~10 (random)  ==
===========================
==  test_arguments       ==
==  difficulty: 0.3      ==
==  size: 10, 15, 20     ==
===========================
"""

args = maze_args
args['size'] = 10
agent = QAgent(in_dim=2, embedding_dim=64)
agent.to(agent.device)
agent.load_state_dict(torch.load('./saved/saved_random_1400.th'))
env = maze_env(args, T=4 * args['size'])

"""evaluation 5 times"""
for eval_id in range(5):
    g, mask = env.reset()
    seq = [env.start_loc]
    while True:
        action = agent.step(g, mask)
        ng, r, n_mask, t = env.step(action)
        seq.append(env.ag_loc)
        agent.push(g, action, mask, r, ng, t)
        g, mask = ng, n_mask
        if t:
            break
    vis_route(args, env.maze, seq, env.start_loc, env.goal_loc, eval_id + 1)
