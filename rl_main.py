import torch

from rl.qagent import QAgent
from env.maze_env import maze_env
from utils.vis_util import vis_route
import wandb

wandb.init(project="etri", entity="curie_ahn")

agent = QAgent(in_dim=4, embedding_dim=64)
agent.to(agent.device)
env = maze_env(grid=5, obstacle_ratio=.1, time_limit=30)

n_ep = 10000

for e in range(n_ep):
    g, mask = env.reset()
    R = 0
    ep_len = 0
    while True:
        ep_len += 1
        action = agent.step(g, mask)
        assert mask.squeeze()[action].item() is False, "{}: maze={}, ag_loc={}, init_loc={}, mask={}".format(ep_len,
                                                                                                             env.maze,
                                                                                                             env.ag_loc,
                                                                                                             env.ag_init_loc,
                                                                                                             mask)
        ng, r, n_mask, t = env.step(action)
        agent.push(g, action, mask, r, ng, t)

        g = ng
        mask = n_mask
        R += r
        if t:
            loss = agent.fit()
            print('EP {}, {} timesteps,  RWD:{:4d}, loss:{:04f}, epsilon:{:05f}'.format(e, ep_len, R, loss,
                                                                                        agent.epsilon))
            wandb.log({"loss": loss, "accum_reward": R, 'ep_len': ep_len, 'epsilon': agent.epsilon,
                       'timestep': ep_len})
            break

    if e % 1000 == 0 and e > 1: torch.save(agent.state_dict(), 'saved_100.th')
