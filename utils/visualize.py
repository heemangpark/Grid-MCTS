import os

import imageio.v2 as imageio
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import natsort
import numpy as np


def frame(maze, start, goal):
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

    ax.plot(start[0] + .5, start[1] + .5, marker='o', markersize=200 / maze.shape[0], color='yellow')
    ax.plot(goal[0] + .5, goal[1] + .5, marker='*', markersize=200 / maze.shape[0], color='yellow')

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

    fig.tight_layout()
    ax.axis('off')

    return fig, ax


def multi_frame(maze, start, goal):
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

    for p in range(len(maze)):
        ax.plot(start[p][0] + .5, start[p][1] + .5, marker='o', markersize=200 / maze[0].shape[0], color='yellow')
        ax.plot(goal[p][0] + .5, goal[p][1] + .5, marker='*', markersize=200 / maze[0].shape[0], color='yellow')

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
    for p in range(len(maze)):
        ax.add_patch(
            patches.Rectangle(
                tuple(start[p]),
                1,
                1,
                facecolor='red',
                fill=True
            ))

        ax.add_patch(
            patches.Rectangle(
                tuple(goal[p]),
                1,
                1,
                facecolor='blue',
                fill=True
            ))

    fig.tight_layout()
    ax.axis('off')

    return fig, ax


def vis_route(maze, seq, start, goal, vis_name):
    fig, ax = frame(maze, start, goal)
    plot_seq = np.array(np.row_stack((start, seq))) + .5
    ax.plot(plot_seq[:, 0], plot_seq[:, 1], linewidth=2, color='k')
    fig.savefig('{}.png'.format(vis_name), bbox_inches='tight', pad_inches=0.1)


def vis_map_only(maze, start, goal, id=0):
    fig, ax = frame(maze, start, goal)
    fig.savefig('maze_{}.png'.format(id), bbox_inches='tight', pad_inches=0.1)


def vis_multi_route(maze, seq, start, goal):
    fig, ax = multi_frame(maze, start, goal)
    plot_seq = [np.array(np.row_stack((start[s], seq[s]))) + .5 for s in range(len(seq))]
    for r in range(len(maze)):
        ax.plot(plot_seq[r][:, 0], plot_seq[r][:, 1], linewidth=2, color='k')
    fig.savefig('{}.png'.format('multi_agent'), bbox_inches='tight', pad_inches=0.1)


def vis_multi_route_total(maze, seq, start, goal):
    fig, ax = multi_frame(maze, start, goal)
    for r in range(len(seq[0])):
        plot_seq = [np.array(np.row_stack((start[m], seq[m][:r + 1]))) + .5 for m in range(len(maze))]
        for i in range(len(maze)):
            ax.plot(plot_seq[i][:, 0], plot_seq[i][:, 1], linewidth=2, color='k')
        fig.savefig('./png_to_gif/{}.png'.format(r + 1), bbox_inches='tight', pad_inches=0.1)

    images = []
    for file_name in natsort.natsorted(os.listdir('./png_to_gif')):
        if file_name.endswith('.png'):
            file_path = os.path.join('./png_to_gif', file_name)
        else:
            pass
        images.append(imageio.imread(file_path))
    imageio.mimsave('./png_to_gif/pathfinding.gif', images, duration=.1)
