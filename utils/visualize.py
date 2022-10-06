import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


def vis_route(maze, seq, start, goal, id=0):
    size = maze.shape[0]
    fig, ax = plt.subplots()

    ax.grid(False)
    ax.set_xticks([i + .5 for i in range(size)])
    ax.set_yticks([i + .5 for i in range(size)])

    ax.set_xticklabels([int(i) for i in range(size)])
    ax.set_yticklabels([int(i) for i in range(size)])
    ax.set_aspect('equal', adjustable='box')

    for i in range(size + 1):
        ax.plot([0, size], [i, i], 'k')

    for j in range(size + 1):
        ax.plot([j, j], [0, size], 'k')

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
    fig.savefig('res_{}.png'.format(id), bbox_inches='tight', pad_inches=0.1)


def vis_map_only(maze, start, goal, id=0):
    fig, ax = plt.subplots()
    size = maze.shape[0]

    ax.grid(False)
    ax.set_xticks([i + .5 for i in range(size)])
    ax.set_yticks([i + .5 for i in range(size)])

    ax.set_xticklabels([int(i) for i in range(size)])
    ax.set_yticklabels([int(i) for i in range(size)])
    ax.set_aspect('equal', adjustable='box')

    for i in range(size + 1):
        ax.plot([0, size], [i, i], 'k')

    for j in range(size + 1):
        ax.plot([j, j], [0, size], 'k')

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
    ax.plot(start[0] + .5, start[1] + .5, marker='o', markersize=7.5, color='yellow')
    ax.plot(goal[0] + .5, goal[1] + .5, marker='*', markersize=15, color='yellow')
    fig.tight_layout()
    fig.savefig('maze_{}.png'.format(id), bbox_inches='tight', pad_inches=0.1)


def vis_route_total(maze, seq, start, goal, id=0):
    size = maze[0].shape[0]
    fig, ax = plt.subplots()

    ax.grid(False)
    ax.set_xticks([i + .5 for i in range(size)])
    ax.set_yticks([i + .5 for i in range(size)])

    ax.set_xticklabels([int(i) for i in range(size)])
    ax.set_yticklabels([int(i) for i in range(size)])
    ax.set_aspect('equal', adjustable='box')

    for i in range(size + 1):
        ax.plot([0, size], [i, i], 'k')

    for j in range(size + 1):
        ax.plot([j, j], [0, size], 'k')

    o_x, o_y = maze[0].nonzero()
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
    for s in range(len(start)):
        ax.add_patch(
            patches.Rectangle(
                tuple(start[s]),
                1,
                1,
                facecolor='red',
                fill=True
            ))
    for g in range(len(goal)):
        ax.add_patch(
            patches.Rectangle(
                tuple(goal[g]),
                1,
                1,
                facecolor='blue',
                fill=True
            ))
    plot_seq = []
    for s in range(len(seq)):
        plot_seq.append(np.array(np.row_stack((start[s], seq[s]))) + .5)
    for p in range(len(maze)):
        ax.plot(plot_seq[p][:, 0], plot_seq[p][:, 1], linewidth=2, color='k')
        ax.plot(start[p][0] + .5, start[p][1] + .5, marker='o', markersize=7.5, color='yellow')
        ax.plot(goal[p][0] + .5, goal[p][1] + .5, marker='*', markersize=15, color='yellow')
    fig.savefig('res_{}.png'.format(id), bbox_inches='tight', pad_inches=0.1)
