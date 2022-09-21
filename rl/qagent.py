import dgl
import torch
import torch.nn as nn

from rl.gnn import GNN
from rl.replaymemory import ReplayMemory


class QAgent(nn.Module):
    def __init__(self, in_dim, embedding_dim):
        super(QAgent, self).__init__()

        self.gnn = GNN(in_dim, out_dim=embedding_dim)
        self.qfunc = nn.Sequential(nn.Linear(embedding_dim, 4),
                                   nn.ReLU())

        self.epsilon = 1.0
        self.batch_size = 100
        self.gamma = .99

        self.memory = ReplayMemory(100000)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def step(self, state, mask, greedy=False):
        self.to('cpu')
        q = self.compute_q(state)
        q[mask] = -999
        if greedy:
            action = q.argmax(-1)
        else:
            random_q = torch.rand_like(q)
            random_q[mask] = -999

            if torch.rand(1) < self.epsilon:
                action = random_q.argmax(-1)
            else:
                action = q.argmax(-1)

            self.epsilon = max(0.05, self.epsilon-0.00001)

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
            nq = self.compute_q(ng)
            nq[mask] = -9999
            n_maxq = nq.max(-1)[0]

        target = r + self.gamma * n_maxq * (1 - t)
        loss = ((target - selected_q) ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
