import pickle
import os
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from fuzz_testing.fuzz_test_smooth_blocking import SmoothBlockingDriver, LaneSwitcherPlanner, LaneSwitcher, \
    SmoothPurePursuitPlanner
from fuzz_testing.fuzz_test_frenet import FrenetDriver, FrenetPlaner, FrenetControllers, FrenetPath


def calculate_coverage(filename):
    'calculate coverage from pickled file'

    try:
        with open(filename, "rb") as f:
            root = pickle.load(f)
            root.coverage.plot()
    except FileNotFoundError:
        pass


def setup_plot():
    fig, ax = plt.subplots()
    ax.set_xlabel('Nodes Explored')
    ax.set_ylabel('Average nearest neighbour metric')

    return ax


def get_all_nodes(root):
    node_lst = [root]
    for c in root.children.values():
        node_lst += get_all_nodes(c)
    return node_lst


def get_crashes(node_lst):
    crashes = 0
    cum_lst = []
    for i in range(0, len(node_lst)):
        if node_lst[i].status == 'error':
            crashes += 1
        cum_lst.append(crashes)
    return cum_lst



def get_max_height(node_lst):
    max_height = 0
    cum_lst = []
    for node in node_lst:
        root = node
        height = 0
        while root:
            root = root.parent
            height += 1
        max_height = max(max_height, height)
        cum_lst.append(max_height)
    return cum_lst


def get_max_distance(node_lst):
    maxim = -np.inf
    minim = np.inf
    cum_lst = []
    for node in node_lst:
        root = node
        maxim = max(maxim, root.obs[1])
        minim = min(minim, root.obs[1])
        cum_lst.append(maxim - minim)
    return cum_lst


def find_dist(p, q, limits_box=None):
    xscale = 1
    yscale = 1
    # print("LIMITS BOX", limits_box)
    if limits_box:
        xscale = limits_box[0][1] - limits_box[0][0]
        yscale = limits_box[1][1] - limits_box[1][0]

    dx = (p[0] - q[0]) / xscale
    dy = (p[1] - q[1]) / yscale
    # print("DX DY")
    # print(p,q)
    # print(dx, dy)
    return np.linalg.norm([dx, dy])


def average_nearest_neighbour(node_lst, nearest_distances):
    # print(node_lst)
    sm = 0
    n = len(node_lst)
    latest_node = node_lst[-1]
    latest_node_nearest_dist = np.inf
    for i in range(n-1):
        node = node_lst[i]
        node_to_latest_node_dist = find_dist(node.obs, latest_node.obs, latest_node.limits_box)
        nearest_distances[i] = min(nearest_distances[i], node_to_latest_node_dist)
        latest_node_nearest_dist = min(latest_node_nearest_dist, node_to_latest_node_dist )
        sm+=nearest_distances[i]
        # print("Part-2 ", i, nearest_distances[i], latest_node_nearest_dist)
    nearest_distances.append(latest_node_nearest_dist)
    sm+=latest_node_nearest_dist
    observed_mean_dist = sm / n
    expected_mean_dist = 0.5 / (sqrt(n))
    # print(observed_mean_dist, expected_mean_dist)
    return observed_mean_dist / expected_mean_dist

def get_cumulative_average_nearest_neighbour(node_lst):
    cum_lst = []
    nearest_distances=[]
    for i in range(len(node_lst)):
        # print(i)
        cum_lst.append(average_nearest_neighbour(node_lst[:i + 1], nearest_distances))
        # average_nearest_neighbour2(node_lst[:i+1],nearest_distances)
    return cum_lst

def get_cumulative_average_nearest_neighbour_collisions(node_lst):
    cum_lst = []
    nearest_distances=[]
    for i in range(len(node_lst)):
        # if node_lst[i].status=='error':
        # print(i)
        cum_lst.append(average_nearest_neighbour(node_lst[:i + 1], nearest_distances))
        # average_nearest_neighbour2(node_lst[:i+1],nearest_distances)
    return cum_lst


def normalize_point(limits_box, p):
    'distance between two points'

    xscale = 1
    yscale = 1

    if limits_box:
        xscale = limits_box[0][1] - limits_box[0][0]
        yscale = limits_box[1][1] - limits_box[1][0]

    dx = (p[0] - limits_box[0][0]) / xscale
    dy = (p[1] - limits_box[1][0]) / yscale

    return [dx, dy]

if __name__ == '__main__':

    cache_dir = "cache/"
    pickled_files = [f for f in os.listdir(cache_dir) if os.path.isfile(os.path.join(cache_dir, f))]
    axis_plot = setup_plot()
    for filename in pickled_files:
        with open(os.path.join(cache_dir, filename), 'rb') as f:
            root_node = pickle.load(f)
            node_lst = get_all_nodes(root_node)
            node_lst.sort(key=lambda x: x.node_timestamp)
            x = np.arange(0, len(node_lst), 1)
            y = get_cumulative_average_nearest_neighbour(node_lst)
            if 'rrt' in filename:
                axis_plot.plot(x, y, 'xkcd:crimson')
            else:
                axis_plot.plot(x, y, 'C4')

    custom_lines = [Line2D([0], [0], color='xkcd:crimson', lw=4),
                    Line2D([0], [0], color='C4', lw=4)]

    axis_plot.legend(custom_lines, ['RRT', 'Random'])
    plt.show()
