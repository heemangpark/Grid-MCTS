import matplotlib.patches as patches
import matplotlib.pyplot as plt

from config import *


def vis_route(tree):
    maze = tree.maze
    seq = tree.state_sequence

    seq = np.array(seq)

    fig, ax = plt.subplots()
    ax.grid(False)
    ax.set_xticks([i + .5 for i in range(MAZE_COLS)])
    ax.set_yticks([i + .5 for i in range(MAZE_ROWS)])

    ax.set_xticklabels([int(i) for i in range(MAZE_COLS)])
    ax.set_yticklabels([int(i) for i in range(MAZE_ROWS)])
    ax.set_aspect('equal', adjustable='box')

    for i in range(MAZE_ROWS + 1):
        ax.plot([0, MAZE_COLS], [i, i], 'black')

    for j in range(MAZE_COLS + 1):
        ax.plot([j, j], [0, MAZE_ROWS], 'black')

    plot_seq = seq + .5
    ax.plot(plot_seq[:, 0], plot_seq[:, 1], 'red', alpha=0.5, linewidth=1)

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
    ax.add_patch(
        patches.Rectangle(
            np.array(FROM[0]),
            1,
            1,
            facecolor='black',
            fill=True
        ))

    ax.add_patch(
        patches.Rectangle(
            np.array(TO),
            1,
            1,
            facecolor='yellow',
            fill=True
        ))

    fig.savefig('res.png')
