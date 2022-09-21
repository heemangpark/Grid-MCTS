import numpy as np

from config import *
from env.maze_func import transition_loc, terminated, get_avail_action


def reward(coordinate):
    if coordinate == FROM[0]:
        return -1
    elif coordinate in OBSTACLES_LINE:
        return -5
    elif (coordinate[0] < 0 or coordinate[0] == MAZE_ROWS) or (coordinate[1] < 0 or coordinate[1] == MAZE_COLS):
        return -5
    else:
        return 0


def children(idx, graph): return list(graph.successors(idx))


def parent(idx, graph): return list(graph.predecessors(idx))[0]


def select(root_idx, graph, c=2):
    score = [graph.edges[edge_idx]['Q'] + c * np.sqrt(
        np.log(1 + graph.nodes[root_idx]['N']) / (1 + graph.edges[edge_idx]['n'])) for edge_idx in
             list(graph.edges(root_idx))]
    if np.var(score) == 0:
        best_child = np.random.choice(children(root_idx, graph))
    else:
        best_child = children(root_idx, graph)[np.argmax(score)]
    best_action = graph.edges[root_idx, best_child]['a']
    return best_child, best_action


def expand(idx, graph, avail_actions):
    leaves = []
    for action in avail_actions:
        next_state = transition_loc(graph.nodes[idx]['state'], action)
        new_child_idx = len(graph) + 1
        leaves.append(new_child_idx)
        graph.add_node(new_child_idx, state=next_state, N=0)
        graph.add_edge(idx, len(graph), Q=0, n=0, a=action, R=0)
    return leaves


def random_rollout(starts_from, steps, map):
    state = starts_from
    ret = 0
    for _ in range(steps):
        action = np.random.choice(get_avail_action(map, state))
        state = transition_loc(state, action)
        ret += terminated(map, state)

    return ret


def backup(leaf_idx, graph):
    while leaf_idx != 1:
        parent_idx = parent(leaf_idx, graph)
        graph.nodes[parent_idx]['N'] += 1
        graph.edges[parent_idx, leaf_idx]['n'] += 1
        ir, sr = reward(graph.nodes[leaf_idx]['state']), random_rollout(graph.nodes[leaf_idx]['state'], 5)
        graph.edges[parent_idx, leaf_idx]['Q'] = 0.2 * ir + 0.8 * sr
        leaf_idx = parent_idx

        if leaf_idx == 1:
            break


def backup_new(leaf_idx, graph, z):
    while leaf_idx != 1:
        parent_idx = parent(leaf_idx, graph)
        graph.nodes[parent_idx]['N'] += 1
        graph.edges[parent_idx, leaf_idx]['n'] += 1
        graph.edges[parent_idx, leaf_idx]['R'] += z
        graph.edges[parent_idx, leaf_idx]['Q'] = graph.edges[parent_idx, leaf_idx]['R'] / \
                                                 graph.edges[parent_idx, leaf_idx]['n']
        leaf_idx = parent_idx

        if leaf_idx == 1:
            break
