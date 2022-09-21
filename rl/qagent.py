from rl.gnn import GNN

import dgl
import torch
import torch.nn as nn


class QAgent(nn.Module):
    def __init__(self, in_dim, embedding_dim):
        super(QAgent, self).__init__()

        self.gnn = GNN(in_dim, out_dim=embedding_dim)
        self.qfunc = nn.Sequential(nn.Linear(embedding_dim, 4),
                                   nn.ReLU())
        self.vfunc = nn.Sequential(nn.Linear(embedding_dim, 1))

        self.memory = Replaymemory(100000)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def step(self, state, greedy=False):
        pass

    def push(self, *args):
        self.memory.push(*args)

    def fit(self):
        if len(self.memory) < self.batch_size:
            return 0

        g, a, mask, r, ng, t = self.memory.sample()

        g = dgl.batch(g)
        ng = dgl.batch(ng)

        q = self.qfunc(g)
        selected_q = q.gather(a, -1)

        with torch.no_grad:
            nq = self.qfunc(ng)
            nq[mask] = -9999
            n_maxq = nq.max(-1)

        target = r + self.gamma * n_maxq * (1 - t)
        loss = ((target - selected_q) ** 2).mean()
