import numpy as np

from utils.arguments import maze_args
from utils.visualize import vis_map_only


def create(args):
    m = np.zeros((args['size'], args['size']))
    o = np.random.random((args['size'], args['size'])) < args['difficulty']
    m[o] = args['cell_type']['obstacle']

    rand_loc = np.random.choice(args['size'], 4)
    if (rand_loc[0] == rand_loc[2]) and (rand_loc[1] == rand_loc[3]):
        while (rand_loc[0] != rand_loc[2]) and (rand_loc[1] != rand_loc[3]):
            rand_loc = np.random.choice(args['size'], 4)

    s = rand_loc[:2]
    g = rand_loc[-2:]
    m[tuple(s)] = args['cell_type']['empty']
    m[tuple(g)] = args['cell_type']['goal']

    return s, g, m


# for id in range(100):
#     """create instance"""
#     sl, gl, maze = create(maze_args)
#
#     """save array data and visualized map"""
#     np.save('../utils/sample_maps/maze_{}'.format(id + 1), maze)
#     np.save('../utils/sample_maps/maze_sg_{}'.format(id + 1), np.array(list(sl), list(gl)))
#     vis_map_only(maze_args, maze, sl, gl, id + 1)
