import torch
import wandb
import random

from env.maze_env import maze_env
from rl.q_agent_loc import QAgent
from utils.arguments import maze_args
from tqdm import tqdm


def main(args, rand=False):
    # wandb.init(project='IoT', entity='heemang')
    wandb.init(project="etri", entity="curie_ahn", config=args)
    agent = QAgent(in_dim=2, embedding_dim=64)
    agent.to(agent.device)
    env = maze_env(args, T=args['size'] * 4)

    n_ep = 100000
    for e in tqdm(range(n_ep)):
        if rand:
            env.args['size'] = random.choice([5, 10, 15, 10])
            env.T = env.args['size'] * 4

        g, mask = env.reset()
        R, ep_len = 0, 0
        while True:
            ep_len += 1
            action = agent.step(g, mask)
            assert mask.squeeze()[action].item() is False, "{}: maze={}, ag_loc={}, init_loc={}, mask={}".format(
                ep_len, env.maze, env.ag_loc, env.start_loc, mask)
            ng, r, n_mask, t = env.step(action)
            agent.push(g, action, mask, r, ng, n_mask, t)

            g, mask = ng, n_mask
            R += r
            if t:
                loss = agent.fit()
                # print('EP {}, {} timesteps,  RWD:{:4d}, loss:{:04f}, epsilon:{:05f}'
                #       .format(e, ep_len, R, loss, agent.epsilon))
                wandb.log({"loss": loss, "reward": R, 'ep_len': ep_len, 'epsilon': agent.epsilon, 'timestep': e})
                break

        if e % 1000 == 0 and e > 0:
            torch.save(agent.state_dict(), './saved/grid_{}_{}.th'.format(args['size'], e))


if __name__ == '__main__':
    main(maze_args)
