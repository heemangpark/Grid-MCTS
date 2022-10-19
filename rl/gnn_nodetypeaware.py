import dgl
import torch
import torch.nn as nn

from functools import partial

OBSTACLE = 1
GOAL = 2
AG = 3


class GNN_nodetypeaware(nn.Module):
    def __init__(self, in_dim, out_dim, embedding_dim=64, n_layers=1, node_types=[0, 2, 3]):
        super(GNN_nodetypeaware, self).__init__()
        in_dims = [in_dim] + [embedding_dim] * (n_layers - 1)
        out_dims = [embedding_dim] * (n_layers - 1) + [out_dim]
        layers = []
        for _i, _o in zip(in_dims, out_dims):
            layers.append(GNNLayer_nodetypeaware(_i, _o, node_types))
        self.layers = nn.ModuleList(layers)

    def forward(self, g, nf):
        for l in self.layers:
            nf = l(g, nf)

        return nf


class GNNLayer_nodetypeaware(nn.Module):
    def __init__(self, in_dim, out_dim, node_types=[0, 2, 3]):
        super(GNNLayer_nodetypeaware, self).__init__()

        self.edge_update = nn.Sequential(nn.Linear(in_dim * 2 + 1, out_dim, bias=False),
                                         nn.ReLU())
        self.node_update = nn.ModuleDict()
        for n in node_types:
            self.node_update['{}'.format(n)] = nn.Sequential(nn.Linear(out_dim + in_dim, out_dim, bias=False),
                                                             nn.ReLU())

        self.node_types = node_types
        self.out_dim = out_dim

    def forward(self, g: dgl.DGLGraph, nf):
        g.ndata['nf'] = nf
        g.send_and_recv(g.edges(), self.message_func, self.reduce_func)

        for ntype in self.node_types:
            filter_func = partial(filter_nodes, ntype=ntype)
            nodes = g.filter_nodes(filter_func)
            apply_func = partial(self.apply_func, ntype=ntype)
            g.apply_nodes(apply_func, v=nodes)

        nf_out = g.ndata.pop('nf_out')

        for k in ['nf']:
            g.ndata.pop(k)

        return nf_out

    def message_func(self, edges):
        src_nf = edges.src['nf']
        dst_nf = edges.dst['nf']
        ef = edges.data['init_ef']

        ef_in = torch.concat([src_nf, dst_nf, ef], -1)
        msg = self.edge_update(ef_in)

        return {'msg': msg, 'ef': ef}

    def reduce_func(self, nodes):
        agg_msg = nodes.mailbox['msg']
        ef = nodes.mailbox['ef'] + 1e-5

        ef_weight = 1 / ef
        normalized_weight = ef_weight / ef_weight.sum(1, keepdims=True)
        agg_msg = agg_msg * normalized_weight

        return {'agg': agg_msg.sum(1)}

    def apply_func(self, nodes, ntype):
        agg_msg = nodes.data['agg']
        nf = nodes.data['nf']
        nf_in = torch.concat([nf, agg_msg], -1)
        nf_out = self.node_update['{}'.format(ntype)](nf_in)
        return {'nf_out': nf_out}


def filter_edges(edges, etype):
    return (edges.data['type'] == etype).squeeze(1)


def filter_nodes(nodes, ntype=AG):
    return (nodes.data['type'] == ntype).squeeze(1)
