import numpy as np
import matplotlib.pyplot as plt
from config import *
import matplotlib.patches as patches


def vis_route(tree):
    maze = tree.maze
    seq = tree.state_sequence

    seq = np.array(seq)

    fig, ax = plt.subplots()
    ax.grid(False)
    ax.set_xticks([i + .5 for i in range(MAZE_COLS)])
    ax.set_yticks([i + .5 for i in range(MAZE_ROWS)])

    ax.set_xticklabels([int(i+1) for i in range(MAZE_COLS)])
    ax.set_yticklabels([int(i+1) for i in range(MAZE_ROWS)])
    ax.set_aspect('equal', adjustable='box')

    for i in range(MAZE_ROWS + 1):
        ax.plot([0, MAZE_COLS], [i, i], 'black')

    for j in range(MAZE_COLS + 1):
        ax.plot([j, j], [0, MAZE_ROWS], 'black')

    plot_seq = seq + 0.5
    ax.plot(plot_seq[:, ], plot_seq[:, 1], 'red')

    for obs in OBSTACLES_LINE:
        ax.add_patch(
            patches.Rectangle(
                obs,
                1,
                1,
                facecolor='red',
                linewidth=2,
                fill=True
            ))
    ax.add_patch(
        patches.Rectangle(
            FROM[0],
            1,
            1,
            facecolor='black',
            fill=True
        ))

    ax.add_patch(
        patches.Rectangle(
            TO,
            1,
            1,
            facecolor='yellow',
            fill=True
        ))

    plt.show()
