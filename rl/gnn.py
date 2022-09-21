import torch
import torch.nn as nn
import dgl


class GNN(nn.Module):
    def __init__(self, in_dim, out_dim, embedding_dim=64, n_layers=1):
        super(GNN, self).__init__()
        in_dims = [in_dim] + [embedding_dim] * n_layers
        out_dims = [embedding_dim] * n_layers + [out_dim]
        layers = []
        for _i, _o in zip(in_dims, out_dims):
            layers.append(GNNLayer(_i, _o))
        self.layers = nn.ModuleList(layers)

    def forward(self, g, nf):
        for l in self.layers:
            nf = l(g, nf)

        return nf


class GNNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GNNLayer, self).__init__()
        self.edge_update = nn.Sequential(nn.Linear(in_dim * 2, out_dim),
                                         nn.ReLU())
        self.node_update = nn.Sequential(nn.Linear(out_dim, out_dim),
                                         nn.ReLU())

    def forward(self, g: dgl.DGLGraph, nf):
        g.ndata['nf'] = nf
        g.update_all(self.message_func, self.reduce_func, self.apply_func)

        # g.ndata.pop('agg')
        # g.edata.pop('msg')
        nf_out = g.ndata.pop('nf_out')
        return nf_out

    def message_func(self, edges):
        src_nf = edges.src['nf']
        dst_nf = edges.dst['nf']

        ef_in = torch.concat([src_nf, dst_nf], -1)
        msg = self.edge_update(ef_in)

        return {'msg': msg}

    def reduce_func(self, nodes):
        msg = nodes.mailbox['msg']
        return {'agg': msg.mean(1)}

    def apply_func(self, nodes):
        nf_in = nodes.data['agg']
        nf_out = self.node_update(nf_in)
        return {'nf_out': nf_out}
