import numpy as np
import random
import os
import json
from collections import deque
import matplotlib.pyplot as plt


def get_trajectories():
    """Get trajectories from the manual control json files.
    and return a tensor of coordinates for the whole batch of trajectories."""
    # read the json files
    files = os.listdir('../../environment/data/lavaenv')
    # sort the files according to name
    files = sorted(files, key=lambda x: int(x.split('_')[-1]))
    # delete the first and last file
    files.pop(0)
    files.pop(-1)
    trajectories = {}
    for file in files:
        with open(f'../../environment/data/lavaenv/{file}/trajectories.json', 'r') as f:
            data = json.load(f)
            #print(file, data)
            trajectories[file] = data['positions']
    return trajectories


def visualize_trajectories(trajectories):
    """Visualize the trajectories."""
    print(trajectories)
    for key, trajectory in trajectories.items():
        x = [point[0] for point in trajectory]
        y = [point[1] for point in trajectory]
        # plot the figure using matplotlib with origin at the top left
        plt.figure()
        plt.plot(x, y)
        plt.xlim(0, 8)
        plt.ylim(0, 8)
        plt.gca().invert_yaxis()
        plt.savefig(f'../../environment/data/lavaenv/{key}/trajectory.png')

 

def main():
    trajectories = get_trajectories()
    visualize_trajectories(trajectories)

if __name__ == "__main__":
    main()