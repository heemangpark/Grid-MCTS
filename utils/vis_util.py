import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt

from config import *


def vis_route(maze, seq):
    seq = np.array(seq)

    ncol = maze.shape[0]
    nrow = maze.shape[1]

    fig, ax = plt.subplots()
    ax.grid(False)
    ax.set_xticks([i + .5 for i in range(ncol)])
    ax.set_yticks([i + .5 for i in range(nrow)])

    ax.set_xticklabels([int(i) for i in range(ncol)])
    ax.set_yticklabels([int(i) for i in range(nrow)])
    ax.set_aspect('equal', adjustable='box')

    for i in range(nrow + 1):
        ax.plot([0, ncol], [i, i], 'black')

    for j in range(ncol + 1):
        ax.plot([j, j], [0, nrow], 'black')

    plot_seq = seq + .5
    ax.plot(plot_seq[:, 0], plot_seq[:, 1], 'red', alpha=1, linewidth=1)

    o_x, o_y = maze.nonzero()

    for i, j in zip(o_x, o_y):
        ax.add_patch(
            patches.Rectangle(
                [i, j],
                1,
                1,
                facecolor='red',
                linewidth=2,
                fill=True
            ))

    i, j = (maze == 2).nonzero()
    # ax.add_patch(
    #     patches.Rectangle(
    #         [i, j],
    #         1,
    #         1,
    #         facecolor='black',
    #         fill=True
    #     ))

    ax.add_patch(
        patches.Rectangle(
            np.array([i, j]),
            1,
            1,
            facecolor='yellow',
            fill=True
        ))

    fig.savefig('res.png')
