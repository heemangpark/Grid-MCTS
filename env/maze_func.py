import numpy as np
import dgl
import torch
from copy import deepcopy

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
move = np.array([[-1, 0], [1, 0], [0, -1], [0, 1], [0, 0]])


def get_avail_action(maze, loc):
    if maze[tuple(loc)] == 2:
        return [4]

    in_bound = []
    size = len(maze)
    if loc[0] != 0:
        in_bound.append(UP)
    if loc[0] != size - 1:
        in_bound.append(DOWN)
    if loc[1] != 0:
        in_bound.append(LEFT)
    if loc[1] != size - 1:
        in_bound.append(RIGHT)

    avail_actions = []
    for a in in_bound:
        new_loc = loc + move[a]
        if maze[tuple(new_loc)] != 1:
            avail_actions.append(a)

    return avail_actions


def transition_loc(loc, action):
    return loc + move[action]


def terminated(goal, loc):
    return tuple(goal) == tuple(loc)


def generate_dense_graph(maze):
    maze = deepcopy(maze)
    m, n = maze.shape
    g = dgl.DGLGraph()
    g.add_nodes(n * m)

    # node position
    rand_x = torch.rand(n - 2).sort().values
    rand_x = torch.concat([torch.Tensor([0]), rand_x, torch.Tensor([1])])

    rand_y = torch.rand(m - 2).sort().values
    rand_y = torch.concat([torch.Tensor([0]), rand_y, torch.Tensor([1])])

    xs = rand_x.repeat(m, 1).reshape(-1)
    ys = rand_y.repeat(n, 1).T.reshape(-1)

    g.ndata['init_nf'] = torch.stack([xs, ys], -1)
    g.ndata['type'] = torch.Tensor(maze.reshape(-1, 1))

    # add edge
    matrix = np.arange(n * m).reshape(n, -1)
    v_from = matrix[:-1].reshape(-1)
    v_to = matrix[1:].reshape(-1)

    g.add_edges(v_from, v_to)
    g.add_edges(v_to, v_from)

    h_from = matrix[:, :-1].reshape(-1)
    h_to = matrix[:, 1:].reshape(-1)

    g.add_edges(h_from, h_to)
    g.add_edges(h_to, h_from)

    dig_from = matrix[:-1, :-1].reshape(-1)
    dig_to = matrix[1:, 1:].reshape(-1)
    g.add_edges(dig_from, dig_to)
    g.add_edges(dig_to, dig_from)

    ddig_from = matrix[1:, :-1].reshape(-1)
    ddig_to = matrix[:-1, 1:].reshape(-1)
    g.add_edges(ddig_from, ddig_to)
    g.add_edges(ddig_to, ddig_from)

    # compute ef
    g.apply_edges(lambda edges: {'init_ef': ((edges.src['init_nf'] - edges.dst['init_nf']) ** 2).sum(-1).reshape(-1, 1) ** .5})

    # remove obstacle
    maze_idx = (maze.reshape(-1) == 1).nonzero()[0]
    g.remove_nodes(maze_idx)

    import matplotlib.pyplot as plt
    import networkx as nx
    G = dgl.to_networkx(g)
    plt.figure()
    pos = {}
    for i in range(g.number_of_nodes()):
        pos[i] = g.ndata['init_nf'][i].tolist()
    nx.draw(G, pos=pos, with_labels=True)
    plt.savefig('example.png')

    return g
