import numpy as np

from env.maze_func import transition_loc


def children(graph, idx): return list(graph.successors(idx))


def parent(graph, idx): return 1 if idx == 1 else list(graph.predecessors(idx))[0]


def select(graph, root_idx, c=2):
    UCT = [graph.nodes[child_idx]['Q'] + c * np.sqrt(
        np.log(graph.nodes[root_idx]['visited']) / (graph.nodes[child_idx]['visited'])) for child_idx in
           children(graph, root_idx)]
    best_child = children(graph, root_idx)[np.argmax(UCT)]
    best_action = graph.edges[root_idx, best_child]['a']
    return best_child, best_action


def expand(graph, idx, avail_actions, tree_type='vanilla'):
    leaves = []
    for a in avail_actions:
        next_state = transition_loc(graph.nodes[idx]['state'], a)
        if tree_type == 'vanilla':
            new_child_idx = len(graph) + 1
            leaves.append(new_child_idx)
            graph.add_node(new_child_idx, state=next_state, visited=0, Q=0)
            graph.add_edge(idx, len(graph), a=['up', 'down', 'left', 'right'][a])
        else:
            if all(next_state == graph.nodes[parent(graph, idx)]['state']):
                pass
            else:
                new_child_idx = len(graph) + 1
                leaves.append(new_child_idx)
                graph.add_node(new_child_idx, state=next_state, visited=0, Q=0)
                graph.add_edge(idx, len(graph), a=['up', 'down', 'left', 'right'][a])
    return leaves


def backup(goal, graph, leaf_idx, reward):
    hop = 0
    while True:
        parent_idx = parent(graph, leaf_idx)
        graph.nodes[parent_idx]['visited'] += 1
        graph.nodes[leaf_idx]['visited'] += 1
        graph.nodes[leaf_idx]['Q'] += (reward * 0.99 ** hop) / graph.nodes[leaf_idx]['visited']
        leaf_idx = parent_idx
        hop += 1
        if leaf_idx == 1:
            graph.nodes[leaf_idx]['visited'] += 1
            graph.nodes[leaf_idx]['Q'] += (reward * 0.99 ** hop) / graph.nodes[leaf_idx]['visited']
            break
