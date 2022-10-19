import random

import torch
import wandb

from env.maze_env_dense import maze_env
from rl.q_agent_densegraph import QAgent
from utils.arguments import maze_args


# from tqdm import tqdm


def main(args, rand=False, exp_name='temp'):
    # wandb.init(project='IoT', entity='heemang')
    # wandb.init(project="etri", entity="curie_ahn", config=args)
    agent = QAgent(in_dim=2)
    agent.to(agent.device)
    env = maze_env(args)

    n_ep = 100000
    best_r = -999
    for e in range(n_ep):
        if rand:
            env.size = random.choice([5, 10, 15, 20])

        g, mask = env.reset()
        R, ep_len = 0, 0
        while True:
            ep_len += 1
            action = agent.step(g, mask)
            ng, r, n_mask, t = env.step(action)
            agent.push(g, action, mask, r, ng, n_mask, t)

            g, mask = ng, n_mask
            R += r
            if t:
                ret_dict = agent.fit()
                exp_dict = {"reward": R, 'ep_len': ep_len, 'epsilon': agent.epsilon, 'episode': e}
                wandb.log({**exp_dict, **ret_dict})
                # print({**exp_dict, **ret_dict})
                break

        if e % 1000 == 0 and e > 0:
            torch.save(agent.state_dict(), './saved/grid_{}_{}.th'.format(exp_name, e))
            # evaluation
            temp_eval_e = 0
            for _ in range(10):
                g, mask = env.reset(size=20)
                R = 0
                while True:
                    action = agent.step(g, mask, greedy=True)
                    g, r, mask, t = env.step(action)
                    R += r
                    if t:
                        temp_eval_e += R
                        break
            temp_eval_e /= 10
            if temp_eval_e > best_r:
                best_r = temp_eval_e
                torch.save(agent.state_dict(), './saved/{}_best_{}.th'.format(exp_name, round(best_r * 100)))


if __name__ == '__main__':
    exp_name = 'rand'
    main(maze_args, rand=True, exp_name=exp_name)
