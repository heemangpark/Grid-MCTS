import numpy as np

from env.maze_func import transition_loc


def distance_score(loc1, loc2):
    dist = -np.sqrt((loc1[0] - loc2[0]) ** 2 + (loc1[1] - loc2[1]) ** 2)
    return 10 if dist > -1 else dist


def children(graph, idx): return list(graph.successors(idx))


def parent(graph, idx): return list(graph.predecessors(idx))[0]


def select(graph, root_idx, c=10):
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
    for action in avail_actions:
        next_state = transition_loc(graph.nodes[idx]['state'], action)
        new_child_idx = len(graph) + 1
        leaves.append(new_child_idx)
        graph.add_node(new_child_idx, state=next_state, N=0)
        graph.add_edge(idx, len(graph), R=0, Q=0, n=0, a=action)
    return leaves


def backup(goal, graph, leaf_idx):
    while leaf_idx != 1:
        leaf_state = graph.nodes[leaf_idx]['state']
        parent_idx = parent(graph, leaf_idx)
        graph.nodes[parent_idx]['N'] += 1
        graph.edges[parent_idx, leaf_idx]['R'] += distance_score(leaf_state, goal)
        graph.edges[parent_idx, leaf_idx]['n'] += 1
        graph.edges[parent_idx, leaf_idx]['Q'] = graph.edges[parent_idx, leaf_idx]['R'] / graph.edges[
            parent_idx, leaf_idx]['n']
        leaf_idx = parent_idx
