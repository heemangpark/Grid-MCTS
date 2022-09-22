import dgl
import torch
import torch.nn as nn

from rl.gnn import GNN
from rl.replaymemory import ReplayMemory
from math import inf


class QAgent(nn.Module):
    def __init__(self, in_dim, embedding_dim):
        super(QAgent, self).__init__()

        self.gnn = GNN(in_dim, out_dim=embedding_dim, n_layers=10)
        self.qfunc = nn.Sequential(nn.Linear(embedding_dim, 4))

        self.gnn_target = GNN(in_dim, out_dim=embedding_dim)
        self.qfunc_target = nn.Sequential(nn.Linear(embedding_dim, 4))

        self.epsilon = 1.0
        self.batch_size = 100
        self.gamma = .9

        self.memory = ReplayMemory(50000)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.update_target(self.gnn, self.gnn_target, 1)
        self.update_target(self.qfunc, self.qfunc_target, 1)

    def step(self, state, mask, greedy=False):
        self.to('cpu')
        q = self.compute_q(state)
        q[mask] = -inf
        if greedy:
            action = q.argmax(-1)
        else:
            random_q = torch.rand_like(q)
            random_q[mask] = -inf

            if torch.rand(1) < self.epsilon:
                action = random_q.argmax(-1)
                self.q = random_q
            else:
                action = q.argmax(-1)
                self.q = q

            self.epsilon = max(0.05, self.epsilon - 0.00005)

        return action.item()

    def compute_q(self, g):
        nf = g.ndata['init_nf']
        updated_nf = self.gnn(g, nf)
        g.ndata['readout'] = updated_nf

        graph_feature = dgl.readout_nodes(g, 'readout')
        qs = self.qfunc(graph_feature)

        g.ndata.pop('readout')
        g.ndata.pop('nf')
        return qs

    def compute_q_target(self, g):
        nf = g.ndata['init_nf']
        updated_nf = self.gnn_target(g, nf)
        g.ndata['readout'] = updated_nf

        graph_feature = dgl.readout_nodes(g, 'readout')
        qs = self.qfunc_target(graph_feature)

        g.ndata.pop('readout')
        g.ndata.pop('nf')
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

        with torch.no_grad():
            nq = self.compute_q_target(ng)
            nq[mask] = -inf
            n_maxq = nq.max(-1)[0]

        target = r + self.gamma * n_maxq * (1 - t)
        loss = ((target - selected_q) ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_target(self.gnn, self.gnn_target, .1)
        self.update_target(self.qfunc, self.qfunc_target, .1)

        return loss.item()

    def update_target(self, src, target, tau):
        for target_param, src_param in zip(target.parameters(), src.parameters()):
            target_param.data.copy_(tau * src_param.data + (1.0 - tau) * target_param.data)
