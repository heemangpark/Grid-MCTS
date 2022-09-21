from rl.qagent import QAgent
from env.maze_env import maze_env

agent = QAgent()
env = maze_env()

n_ep = 10000

for e in range(n_ep):
    g, mask = env.reset()
    while True:
        action = agent.get_action(g, mask)
        ng, r, n_mask, t = env.step(action)
        agent.push(g, action, mask, r, ng, t)

        g = ng
        mask = n_mask
        if t:
            loss = agent.fit()
            break
