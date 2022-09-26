import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


def vis_route(args, maze, seq, start, goal, id):
    fig, ax = plt.subplots()

    ax.grid(False)
    ax.set_xticks([i + .5 for i in range(args['size'])])
    ax.set_yticks([i + .5 for i in range(args['size'])])

    ax.set_xticklabels([int(i) for i in range(args['size'])])
    ax.set_yticklabels([int(i) for i in range(args['size'])])
    ax.set_aspect('equal', adjustable='box')

    for i in range(args['size'] + 1):
        ax.plot([0, args['size']], [i, i], 'k')

    for j in range(args['size'] + 1):
        ax.plot([j, j], [0, args['size']], 'k')

    o_x, o_y = maze.nonzero()
    for i, j in zip(o_x, o_y):
        ax.add_patch(
            patches.Rectangle(
                (i, j),
                1,
                1,
                facecolor='green',
                linewidth=2,
                fill=True
            ))
    ax.add_patch(
        patches.Rectangle(
            tuple(start),
            1,
            1,
            facecolor='red',
            fill=True
        ))

    ax.add_patch(
        patches.Rectangle(
            tuple(goal),
            1,
            1,
            facecolor='blue',
            fill=True
        ))

    plot_seq = np.array(np.row_stack((start, seq))) + .5
    ax.plot(plot_seq[:, 0], plot_seq[:, 1], linewidth=2, color='k')
    ax.plot(start[0] + .5, start[1] + .5, marker='o', markersize=7.5, color='yellow')
    ax.plot(goal[0] + .5, goal[1] + .5, marker='*', markersize=15, color='yellow')
    fig.savefig('res_{}.png'.format(id))
