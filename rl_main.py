import torch

from rl.qagent import QAgent
from env.maze_env import maze_env
from utils.vis_util import vis_route

agent = QAgent(in_dim=4, embedding_dim=64)
agent.to(agent.device)
env = maze_env()

n_ep = 10000

for e in range(n_ep):
    g, mask = env.reset()
    R = 0
    while True:
        action = agent.step(g, mask)
        ng, r, n_mask, t = env.step(action)
        agent.push(g, action, mask, r, ng, t)

        g = ng
        mask = n_mask
        R += r
        if t:
            loss = agent.fit()
            print('{}th EP, RWD:{:5d}, loss:{:05f}, epsilon:{:05f}'.format(e, R, loss, agent.epsilon))
            break

torch.save(agent.state_dict(), 'saved.th')
