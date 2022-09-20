import numpy as np

from config import *


def condition(coordinate):
    return False if (coordinate[0] < 0 or coordinate[0] == MAZE_ROWS) or (
            coordinate[1] < 0 or coordinate[1] == MAZE_COLS) else False if coordinate in OBSTACLES_LINE else True


def transition(state, action):
    if action == "up":
        temp_state = [state[0] - 1, state[1]]
        if condition(temp_state):
            state = temp_state
    elif action == "down":
        temp_state = [state[0] + 1, state[1]]
        if condition(temp_state):
            state = temp_state
    elif action == "left":
        temp_state = [state[0], state[1] - 1]
        if condition(temp_state):
            state = temp_state
    else:
        temp_state = [state[0], state[1] + 1]
        if condition(temp_state):
            state = temp_state
    return state


def distance_score(loc1, loc2): return -np.sqrt((loc1[0] - loc2[0]) ** 2 + (loc1[1] - loc2[1]) ** 2)


def children(idx, graph): return list(graph.successors(idx))


def parent(idx, graph): return list(graph.predecessors(idx))[0]


def select(root_idx, graph, c=2):
    score = [(graph.edges[edge_idx]['Q']) + c * np.sqrt(np.log(graph.nodes[root_idx]['N']) / graph.edges[edge_idx]['n'])
             for edge_idx in list(graph.edges(root_idx))]
    if np.var(score) == 0:
        best_child = np.random.choice(children(root_idx, graph))
    else:
        best_child = children(root_idx, graph)[np.argmax(score)]
    best_action = graph.edges[root_idx, best_child]['a']
    return best_child, best_action


def expand(idx, graph, setting='single'):
    leaves = []
    if setting == 'single':
        actions = ['up', 'down', 'left', 'right']
    else:
        actions = [[first, second] for first in ['up', 'down', 'left', 'right'] for second in
                   ['up', 'down', 'left', 'right']]
    for action in actions:
        next_state = transition(graph.nodes[idx]['state'], action)
        new_child_idx = len(graph) + 1
        leaves.append(new_child_idx)
        graph.add_node(new_child_idx, state=next_state, N=1)
        graph.add_edge(idx, len(graph), Q=0, n=1, a=action)
    return leaves


def simulated_reward(starts_from, steps=10, sim_num=100):
    reward_list = []
    for _ in range(sim_num):
        state, reward = starts_from, 0
        for _ in range(steps):
            action = np.random.choice(['up', 'down', 'left', 'right'])
            next_state = transition(state, action)
            if next_state == TO:
                reward += 10
            elif next_state in OBSTACLES_LINE:
                reward -= 10
            else:
                pass
            state = next_state
        reward_list.append(reward)
    return np.mean(reward_list)


def backup(leaf_idx, graph):
    while leaf_idx != 1:
        leaf_coord = graph.nodes[leaf_idx]['state']
        r, sr = distance_score(leaf_coord, TO), simulated_reward(leaf_coord)
        parent_idx = parent(leaf_idx, graph)
        graph.nodes[parent_idx]['N'] += 1
        graph.edges[parent_idx, leaf_idx]['n'] += 1
        graph.edges[parent_idx, leaf_idx]['Q'] += (r + sr) / 2
        leaf_idx = parent_idx
