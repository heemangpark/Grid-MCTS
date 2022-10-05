from env.maze_func import transition_loc
from itertools import product
from mcts_nn.tree_functions import parent


def expand_joint(graph, idx, joint_avail_actions):
    leaves = []
    for joint_action in product(*joint_avail_actions):
        next_joint_state = []
        for i, a in enumerate(joint_action):
            next_state = transition_loc(graph.nodes[idx]['state'][i], a)
            next_joint_state.append(list(next_state))
        new_child_idx = len(graph) + 1
        leaves.append(new_child_idx)
        graph.add_node(new_child_idx, state=next_joint_state, visited=0, Q=0)
        graph.add_edge(idx, len(graph), a=[['up', 'down', 'left', 'right'][action] for action in joint_action])
    return leaves


def backup(graph, leaf_idx, r_value, leaf_maxq):
    hop = 0
    while True:
        parent_idx = parent(graph, leaf_idx)
        graph.nodes[parent_idx]['visited'] += 1
        graph.nodes[leaf_idx]['visited'] += 1
        graph.nodes[leaf_idx]['Q'] += (r_value + leaf_maxq.max() * 0.99 ** hop) / graph.nodes[leaf_idx]['visited']
        # graph.edges[parent_idx, leaf_idx]['R'] += q_value.mean() * 0.99 ** hop
        # graph.edges[parent_idx, leaf_idx]['n'] += 1
        # graph.edges[parent_idx, leaf_idx]['Q'] = graph.edges[parent_idx, leaf_idx]['R'] / \
        #                                          graph.edges[parent_idx, leaf_idx]['n']
        leaf_idx = parent_idx
        hop += 1
        if leaf_idx == 1:
            graph.nodes[leaf_idx]['visited'] += 1
            graph.nodes[leaf_idx]['Q'] += (r_value + leaf_maxq.max() * 0.99 ** hop) / graph.nodes[leaf_idx]['visited']
            break
