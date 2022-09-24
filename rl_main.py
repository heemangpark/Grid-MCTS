import torch

import wandb
from env.maze_env import maze_env
from rl.qagent import QAgent
from utils.arguments import get_random_maze_args

# wandb.init(project="etri", entity="curie_ahn")
# wandb.init(project='IoT', entity='heemang')

agent = QAgent(in_dim=4, embedding_dim=64)
agent.to(agent.device)
env = maze_env(get_random_maze_args(), time_limit=30)

n_ep = 100000
for e in range(n_ep):
    g, mask = env.reset()
    R, ep_len = 0, 0
    while True:
        ep_len += 1
        action = agent.step(g, mask)
        assert mask.squeeze()[action].item() is False, \
            "{}: maze={}, ag_loc={}, init_loc={}, mask={}".format(ep_len, env.maze, env.ag_loc, env.init_loc, mask)
        ng, r, n_mask, t = env.step(action)
        agent.push(g, action, mask, r, ng, t)

        g, mask = ng, n_mask
        R += r
        if t:
            loss = agent.fit()
            print('EP {}, {} timesteps,  RWD:{:4d}, loss:{:04f}, epsilon:{:05f}'
                  .format(e, ep_len, R, loss, agent.epsilon))
            # wandb.log({"loss": loss, "accum_reward": R, 'ep_len': ep_len, 'epsilon': agent.epsilon, 'timestep': e})
            break

    if e % 1000 == 0 and e > 0:
        torch.save(agent.state_dict(), './saved/saved_{}.th'.format(e))
