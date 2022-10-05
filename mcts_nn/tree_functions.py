import numpy as np

from env.maze_func import transition_loc


def distance_score(loc1, loc2):
    dist = -np.sqrt((loc1[0] - loc2[0]) ** 2 + (loc1[1] - loc2[1]) ** 2)
    return 10 if dist > -1 else dist


def mask4tree(env):
    mask = []
    move = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
    for a in [0, 1, 2, 3]:  # 상 하 좌 우
        if (0 <= (env.ag_loc + move[a])[0] < env.size) and (
                0 <= (env.ag_loc + move[a])[1] < env.size):
            if env.maze[tuple(env.ag_loc + move[a])] == 0:
                m = True
            elif env.maze[tuple(env.ag_loc + move[a])] == 1:
                m = False
            elif env.maze[tuple(env.ag_loc + move[a])] == 2:
                m = True
            else:
                m = None
        else:
            m = False
        mask.append(m)

    return mask


def children(graph, idx): return list(graph.successors(idx))


def parent(graph, idx): return list(graph.predecessors(idx))[0]


def select(graph, root_idx, c=2):
    # score = [(graph.edges[edge_idx]['R']) + c * np.sqrt(
    #     np.log(graph.nodes[root_idx]['N'] + 1e-4) / (graph.edges[edge_idx]['n'] + 1)) for edge_idx in
    #          list(graph.edges(root_idx))]
    UCT = [graph.nodes[child_idx]['Q'] + c * np.sqrt(
        np.log(graph.nodes[root_idx]['visited']) / (graph.nodes[child_idx]['visited'])) for child_idx in
           children(graph, root_idx)]
    # if np.var(UCT) == 0:
    #     best_child = np.random.choice(children(graph, root_idx))
    # else:
    best_child = children(graph, root_idx)[np.argmax(UCT)]
    best_action = graph.edges[root_idx, best_child]['a']
    return best_child, best_action


def expand(graph, idx, avail_actions):
    leaves = []
    for a in avail_actions:
        next_state = transition_loc(graph.nodes[idx]['state'], a)
        new_child_idx = len(graph) + 1
        leaves.append(new_child_idx)
        graph.add_node(new_child_idx, state=next_state, visited=0, Q=0)
        graph.add_edge(idx, len(graph), a=['up', 'down', 'left', 'right'][a])
    return leaves


def backup(graph, leaf_idx, r_value, q_value):
    hop = 0
    while True:
        parent_idx = parent(graph, leaf_idx)
        graph.nodes[parent_idx]['visited'] += 1
        graph.nodes[leaf_idx]['visited'] += 1
        graph.nodes[leaf_idx]['Q'] += (r_value + q_value.max() * 0.99 ** hop) / graph.nodes[leaf_idx]['visited']
        # graph.edges[parent_idx, leaf_idx]['R'] += q_value.mean() * 0.99 ** hop
        # graph.edges[parent_idx, leaf_idx]['n'] += 1
        # graph.edges[parent_idx, leaf_idx]['Q'] = graph.edges[parent_idx, leaf_idx]['R'] / \
        #                                          graph.edges[parent_idx, leaf_idx]['n']
        leaf_idx = parent_idx
        hop += 1
        if leaf_idx == 1:
            graph.nodes[leaf_idx]['visited'] += 1
            graph.nodes[leaf_idx]['Q'] += (r_value + q_value.max() * 0.99 ** hop) / graph.nodes[leaf_idx]['visited']
            break
