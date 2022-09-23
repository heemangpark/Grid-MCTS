import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


def vis_route(args, maze, seq, agent_id):
    seq = np.array(np.row_stack((np.array(args.start[agent_id - 1]), seq)))
    fig, ax = plt.subplots()

    ax.grid(False)
    ax.set_xticks([i + .5 for i in range(args.maze_x)])
    ax.set_yticks([i + .5 for i in range(args.maze_y)])

    ax.set_xticklabels([int(i) for i in range(args.maze_x)])
    ax.set_yticklabels([int(i) for i in range(args.maze_y)])
    ax.set_aspect('equal', adjustable='box')

    for i in range(args.maze_y + 1):
        ax.plot([0, args.maze_x], [i, i], 'k')

    for j in range(args.maze_x + 1):
        ax.plot([j, j], [0, args.maze_y], 'k')

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
            args.start[agent_id - 1],
            1,
            1,
            facecolor='red',
            fill=True
        ))

    ax.add_patch(
        patches.Rectangle(
            args.goal,
            1,
            1,
            facecolor='blue',
            fill=True
        ))

    plot_seq = seq + .5
    ax.plot(plot_seq[:, 0], plot_seq[:, 1], linewidth=2, color='k')
    ax.plot(plot_seq[0][0], plot_seq[0][1], marker='o', markersize=7.5, color='yellow')
    ax.plot(plot_seq[-1][0], plot_seq[-1][1], marker='*', markersize=15, color='yellow')
    fig.savefig('res.png')
