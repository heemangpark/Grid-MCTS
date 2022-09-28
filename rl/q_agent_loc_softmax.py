from math import inf

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.gnn_typeaware import GNN_typeaware, filter_ag_nodes
from rl.replaymemory import ReplayMemory


class QAgent(nn.Module):
    def __init__(self, in_dim, embedding_dim):
        super(QAgent, self).__init__()
        self.q = None

        self.gnn = GNN_typeaware(in_dim, out_dim=embedding_dim, n_layers=2)
        self.q_func = nn.Sequential(nn.Linear(embedding_dim, embedding_dim / 2),
                                    nn.ReLU(),
                                    nn.Linear(embedding_dim, 4)
                                    )

        self.gnn_target = GNN_typeaware(in_dim, out_dim=embedding_dim, n_layers=2)
        self.q_func_target = nn.Sequential(nn.Linear(embedding_dim, embedding_dim / 2),
                                           nn.ReLU(),
                                           nn.Linear(embedding_dim, 4)
                                           )

        self.epsilon = 1.0
        self.batch_size = 100
        self.gamma = .95

        self.memory = ReplayMemory(10000)
        self.optimizer = torch.optim.Adam(list(self.gnn.parameters()) + list(self.q_func.parameters()), lr=1e-3)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.update_target(self.gnn, self.gnn_target, 1)
        self.update_target(self.q_func, self.q_func_target, 1)

    def step(self, state, mask, greedy=False):
        self.to('cpu')
        q = self.compute_q_target(state)
        q[mask] = -inf

        if greedy:
            action = q.argmax(-1)
        else:
            q_logit = F.log_softmax(q, dim=0)
            action_dist = torch.distributions.Categorical(logits=q_logit)
            action = action_dist.sample()

        return action.item()

    def compute_q(self, g):
        nf = g.ndata['init_nf']
        updated_nf = self.gnn(g, nf)
        ag_nodes = g.filter_nodes(filter_ag_nodes)
        ag_nf = updated_nf[ag_nodes]
        qs = self.q_func(ag_nf)

        return qs

    def compute_q_target(self, g):
        nf = g.ndata['init_nf']
        updated_nf = self.gnn_target(g, nf)
        ag_nodes = g.filter_nodes(filter_ag_nodes)
        ag_nf = updated_nf[ag_nodes]
        qs = self.q_func_target(ag_nf)

        return qs

    def push(self, *args):
        self.memory.push(*args)

    def fit(self):
        if len(self.memory) < self.batch_size:
            return 0
        self.to(self.device)
        g, a, mask, r, ng, t = self.memory.sample(self.batch_size)

        g = dgl.batch(g).to(self.device)
        ng = dgl.batch(ng).to(self.device)
        a = torch.tensor(a).to(self.device)
        r = torch.Tensor(r).to(self.device)
        t = torch.tensor(t).to(self.device) + 0
        mask = torch.stack(mask).squeeze()

        q = self.compute_q(g)
        selected_q = q.gather(-1, a.reshape(-1, 1))
        selected_q = selected_q.squeeze()
        norm_selected_q = (selected_q - torch.mean(selected_q)) / torch.std(selected_q)

        with torch.no_grad():
            nq = self.compute_q_target(ng)
            nq[mask] = -inf
            n_max_q = nq.max(-1)[0]

        target = r + self.gamma * n_max_q * (1 - t)
        norm_target = (target - torch.mean(target)) / torch.std(target)
        # loss = ((target - selected_q) ** 2).mean()
        loss = ((norm_target - norm_selected_q) ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_target(self.gnn, self.gnn_target, .9)
        self.update_target(self.q_func, self.q_func_target, .9)

        return loss.item()

    def update_target(self, src, target, tau):
        for target_param, src_param in zip(target.parameters(), src.parameters()):
            target_param.data.copy_(tau * src_param.data + (1.0 - tau) * target_param.data)
