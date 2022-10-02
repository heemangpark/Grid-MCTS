import random

import torch

import wandb
from env.maze_env import maze_env
from rl.q_agent_loc import QAgent
from utils.arguments import maze_args

# wandb.init(project='IoT', entity='heemang')
# wandb.init(project="etri", entity="curie_ahn", config=maze_args)

args = maze_args
agent = QAgent(in_dim=2, embedding_dim=64)
agent.to(agent.device)

n_ep = 100000
for e in range(n_ep):
    n_grid = random.choice([5, 6, 7, 8, 9, 10])
    args['size'] = n_grid
    env = maze_env(args, T=n_grid * 4)
    g, mask = env.reset()
    R, ep_len = 0, 0
    while True:
        ep_len += 1
        action = agent.step(g, mask)
        assert mask.squeeze()[action].item() is False, \
            "{}: maze={}, ag_loc={}, init_loc={}, mask={}".format(ep_len, env.maze, env.ag_loc, env.start_loc, mask)
        ng, r, n_mask, t = env.step(action)
        agent.push(g, action, mask, r, ng, t)

        g, mask = ng, n_mask
        R += r
        if t:
            loss = agent.fit()
            print('EP {}, {} timesteps,  RWD:{:4d}, loss:{:04f}, epsilon:{:05f}'
                  .format(e, ep_len, R, loss, agent.epsilon))
            # wandb.log({"loss": loss, "reward": R, 'ep_len': ep_len, 'epsilon': agent.epsilon, 'timestep': e})
            break

    if e % 1000 == 0 and e > 0:
        torch.save(agent.state_dict(), './saved/saved_random_{}.th'.format(e))
