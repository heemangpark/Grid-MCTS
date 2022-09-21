import numpy as np
from env.maze_func import transition_loc, terminated
from config import TO


def distance_score(loc1, loc2):
    dist = -np.sqrt((loc1[0] - loc2[0]) ** 2 + (loc1[1] - loc2[1]) ** 2)
    if dist > -2:
        return 10
    return dist


def children(idx, graph): return list(graph.successors(idx))


def parent(idx, graph): return list(graph.predecessors(idx))[0]


def select(root_idx, graph, c=2):
    score = [
        (graph.edges[edge_idx]['Q']) + c * np.sqrt(
            np.log(graph.nodes[root_idx]['N'] + 1e-4) / (graph.edges[edge_idx]['n'] + 1))
        for edge_idx in list(graph.edges(root_idx))]
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


# def simulated_reward(starts_from, steps=10, sim_num=100):
#     reward_list = []
#     for _ in range(sim_num):
#         state, reward = starts_from, 0
#         for _ in range(steps):
#             action = np.random.choice(['up', 'down', 'left', 'right'])
#             next_state = transition_loc(state, action)
#             if next_state == TO:
#                 reward += 10
#             elif next_state in OBSTACLES_LINE:
#                 reward -= 10
#             else:
#                 pass
#             state = next_state
#         reward_list.append(reward)
#     return np.mean(reward_list)


def backup(leaf_idx, graph):
    while leaf_idx != 1:
        leaf_coord = graph.nodes[leaf_idx]['state']
        r = distance_score(leaf_coord, TO)
        parent_idx = parent(leaf_idx, graph)
        graph.nodes[parent_idx]['N'] += 1
        graph.edges[parent_idx, leaf_idx]['n'] += 1
        graph.edges[parent_idx, leaf_idx]['R'] += r
        graph.edges[parent_idx, leaf_idx]['Q'] = graph.edges[parent_idx, leaf_idx]['R'] / \
                                                 graph.edges[parent_idx, leaf_idx]['n']
        leaf_idx = parent_idx
