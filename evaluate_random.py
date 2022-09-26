from env.maze_env import maze_env
from rl.q_agent_loc import QAgent
from utils.arguments import maze_args
from utils.visualize import vis_route

"""train: 5~10 -> test: 10"""
args = maze_args
agent = QAgent(in_dim=2, embedding_dim=64)
agent.to(agent.device)
env = maze_env(args, T=40)

"""evaluation"""
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
vis_route(args, env.maze, seq, env.start_loc, env.goal_loc)
