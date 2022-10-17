from copy import deepcopy
from itertools import product

import numpy as np

from env.maze_func import transition_loc


def mask4tree(maze, loc):
    mask = []
    size = maze.shape[0]
    move = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
    for a in [0, 1, 2, 3]:
        if (0 <= (loc + move[a])[0] < size) and (
                0 <= (loc + move[a])[1] < size):
            if maze[tuple(loc + move[a])] == 0:
                m = True
            elif maze[tuple(loc + move[a])] == 1:
                m = False
            elif maze[tuple(loc + move[a])] == 2:
                m = True
            else:
                m = None
        else:
            m = False
        mask.append(m)

    return mask


def children(graph, idx):
    return list(graph.successors(idx))


def parent(graph, idx):
    return 1 if idx == 1 else list(graph.predecessors(idx))[0]


def select(graph, root_idx, c=2):
    UCT = [graph.nodes[child_idx]['Q'] + c * np.sqrt(
        np.log(graph.nodes[root_idx]['visited']) / (graph.nodes[child_idx]['visited'])) for child_idx in
           children(graph, root_idx)]
    best_child = children(graph, root_idx)[np.argmax(UCT)]
    best_action = graph.edges[root_idx, best_child]['a']
    return best_child, best_action


def expand_joint(graph, idx, joint_avail_actions, tree_type='vanilla'):
    leaves = []
    prev_maze = graph.nodes[idx]['maze']
    for joint_action in product(*joint_avail_actions):
        next_joint_state = []
        new_maze = deepcopy(prev_maze)
        for i, a in enumerate(joint_action):
            next_state = transition_loc(graph.nodes[idx]['state'][i], a)
            next_joint_state.append(list(next_state))
            'ergodic maze'
            if new_maze[i][tuple(next_state)] == 2:
                pass
            else:
                new_maze[i][tuple(next_state)] = 1

        if tree_type == 'vanilla':
            new_child_idx = len(graph) + 1
            leaves.append(new_child_idx)
            graph.add_node(new_child_idx, state=next_joint_state, visited=0, Q=0, maze=new_maze)
            graph.add_edge(idx, len(graph), a=[['up', 'down', 'left', 'right', 't'][action] for action in joint_action])

        elif tree_type == 'grand':
            check_bool = [next_joint_state[i] == graph.nodes[parent(graph, idx)]['state'][i] for i in
                          range(len(next_joint_state))]
            if not any(check_bool):
                new_child_idx = len(graph) + 1
                leaves.append(new_child_idx)
                graph.add_node(new_child_idx, state=next_joint_state, visited=0, Q=0, maze=new_maze)
                graph.add_edge(idx, len(graph),
                               a=[['up', 'down', 'left', 'right', 't'][action] for action in joint_action])

    return leaves


def backup(graph, leaf_idx, leaf_r, leaf_v):
    hop = 0
    while True:
        parent_idx = parent(graph, leaf_idx)
        graph.nodes[parent_idx]['visited'] += 1
        graph.nodes[leaf_idx]['visited'] += 1
        graph.nodes[leaf_idx]['Q'] += (leaf_r + leaf_v) * 0.99 ** hop
        leaf_idx = parent_idx
        hop += 1

        if leaf_idx == 1:
            graph.nodes[leaf_idx]['visited'] += 1
            graph.nodes[leaf_idx]['Q'] += (leaf_r + leaf_v) * 0.99 ** hop
            break
