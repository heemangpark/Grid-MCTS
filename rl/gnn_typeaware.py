import dgl
import torch
import torch.nn as nn

from functools import partial

OBSTACLE = 1
GOAL = 2
AG = 3


class GNN_typeaware(nn.Module):
    def __init__(self, in_dim, out_dim, embedding_dim=64, n_layers=1, edge_types=[1, 2, 3]):
        super(GNN_typeaware, self).__init__()
        in_dims = [in_dim] + [embedding_dim] * (n_layers - 1)
        out_dims = [embedding_dim] * (n_layers - 1) + [out_dim]
        layers = []
        for _i, _o in zip(in_dims, out_dims):
            layers.append(GNNLayer_typeaware(_i, _o, edge_types))
        self.layers = nn.ModuleList(layers)

    def forward(self, g, nf):
        for l in self.layers:
            nf = l(g, nf)

        return nf


class GNNLayer_typeaware(nn.Module):
    def __init__(self, in_dim, out_dim, edge_types=[1, 2, 3]):
        super(GNNLayer_typeaware, self).__init__()

        self.edge_update = nn.ModuleDict()
        for etype in edge_types:
            self.edge_update[str(etype)] = nn.Sequential(nn.Linear(in_dim * 2, out_dim, bias=False),
                                                         nn.ReLU())

        self.node_update = nn.Sequential(nn.Linear(out_dim * len(edge_types), out_dim, bias=False),
                                         nn.ReLU())
        self.edge_types = edge_types
        self.out_dim = out_dim

    def forward(self, g: dgl.DGLGraph, nf):
        g.ndata['nf'] = nf

        for i, etype in enumerate(self.edge_types):
            filter_func = partial(filter_edges, etype=etype)
            message_func = partial(self.message_func, etype=etype)
            reduce_func = partial(self.reduce_func, etype=etype)
            g.ndata['agg{}'.format(etype)] = torch.zeros((g.number_of_nodes(), self.out_dim)).to(nf.device)

            edges = g.filter_edges(filter_func)
            g.send_and_recv(edges, message_func, reduce_func)

        ag_nodes = g.filter_nodes(filter_ag_nodes)
        g.apply_nodes(self.apply_func, v=ag_nodes)

        nf_out = g.ndata.pop('nf_out')

        for k in ['nf', 'agg1', 'agg2', 'agg3']:
            g.ndata.pop(k)

        return nf_out

    def message_func(self, edges, etype):
        src_nf = edges.src['nf']
        dst_nf = edges.dst['nf']

        ef_in = torch.concat([src_nf, dst_nf], -1)
        msg = self.edge_update[str(etype)](ef_in)

        return {'msg{}'.format(etype): msg}

    def reduce_func(self, nodes, etype):
        agg_msg = nodes.mailbox['msg{}'.format(etype)]

        return {'agg{}'.format(etype): agg_msg.mean(1)}

    def apply_func(self, nodes):
        msgs = []
        for e in self.edge_types:
            msgs.append(nodes.data['agg{}'.format(e)])
        nf_in = torch.concat(msgs, -1)
        nf_out = self.node_update(nf_in)
        return {'nf_out': nf_out}


def filter_edges(edges, etype):
    return (edges.data['type'] == etype).squeeze(1)


def filter_ag_nodes(nodes, ntype=AG):
    return (nodes.data['type'] == ntype).squeeze(1)
