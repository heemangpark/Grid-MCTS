from copy import deepcopy

import numpy as np
import torch

from env.maze_func import transition_loc


def distance_score(loc1, loc2):
    dist = -np.sqrt((loc1[0] - loc2[0]) ** 2 + (loc1[1] - loc2[1]) ** 2)
    return 10 if dist > -1 else dist


def convert_maze_to_g_loc(ag_loc, base_graph):
    g = deepcopy(base_graph)
    g.add_nodes(1)
    n_nodes = g.number_of_nodes()
    g.ndata['init_nf'][-1] = torch.Tensor(ag_loc) / 20
    g.add_edges(range(n_nodes), n_nodes - 1)
    g.ndata['type'] = torch.Tensor([1] * (n_nodes - 2) + [2] + [3]).reshape(-1, 1)
    g.edata['type'] = torch.Tensor([1] * (n_nodes - 2) + [2] + [3]).reshape(-1, 1)

    return g


def mask(ag_loc, maze):
    mask = []
    move = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
    for a in [0, 1, 2, 3]:  # 상 하 좌 우
        if (0 <= list(ag_loc + move[a])[0] < 20) and (
                0 <= list(ag_loc + move[a])[1] < 20):
            if maze[tuple(ag_loc + move[a])] == 0:
                m = True
            elif maze[tuple(ag_loc + move[a])] == 1:
                m = False
            elif maze[tuple(ag_loc + move[a])] == 2:
                m = True
            else:
                m = None
        else:
            m = False
        mask.append(m)

    return mask


def children(graph, idx): return list(graph.successors(idx))


def parent(graph, idx): return list(graph.predecessors(idx))[0]


def select(graph, root_idx, c=1):
    score = [(graph.edges[edge_idx]['R']) + c * np.sqrt(
        np.log(graph.nodes[root_idx]['N'] + 1e-4) / (graph.edges[edge_idx]['n'] + 1)) for edge_idx in
             list(graph.edges(root_idx))]
    if np.var(score) == 0:
        best_child = np.random.choice(children(graph, root_idx))
    else:
        best_child = children(graph, root_idx)[np.argmax(score)]
    best_action = graph.edges[root_idx, best_child]['a']
    return best_child, best_action


def expand(graph, idx, avail_actions):
    leaves = []
    for a in avail_actions:
        next_state = transition_loc(graph.nodes[idx]['state'], a)
        new_child_idx = len(graph) + 1
        leaves.append(new_child_idx)
        graph.add_node(new_child_idx, state=next_state, N=0)
        graph.add_edge(idx, len(graph), R=0, Q=0, n=0, a=['up', 'down', 'left', 'right'][a])
    return leaves


def backup(agent, base_graph, graph, leaf_idx):
    while leaf_idx != 1:
        leaf_state = list(graph.nodes[leaf_idx]['state'])
        parent_idx = parent(graph, leaf_idx)
        graph.nodes[parent_idx]['N'] += 1
        q = agent.step(convert_maze_to_g_loc(leaf_state, base_graph), [True for _ in range(4)], tree_search=True)
        graph.edges[parent_idx, leaf_idx]['R'] += q.mean()
        graph.edges[parent_idx, leaf_idx]['n'] += 1
        graph.edges[parent_idx, leaf_idx]['Q'] = graph.edges[parent_idx, leaf_idx]['R'] / graph.edges[parent_idx,
                                                                                                      leaf_idx]['n']
        leaf_idx = parent_idx
