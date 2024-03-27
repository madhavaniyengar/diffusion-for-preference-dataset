import numpy as np
import random
import os
import json
from collections import deque
import matplotlib.pyplot as plt
import torch


def get_trajectories():
    """Get trajectories from the manual control json files.
    and return a tensor of coordinates for the whole batch of trajectories."""
    # read the json files
    # relative_path = '../../environment/data/lavaenv'
    # absolute_path = '/teamspace/studios/this_studio/diffusion-features/environment/data/lavaenv'
    # absolute_path = '/Users/sagarpatil/sagar/projects/diffusion-features/environment/data/lavaenv'
    absolute_path = '/home/miyen/diffusion-features/environment/data/lavaenv'
    print(absolute_path)
    files = os.listdir(absolute_path)
    # sort the files according to name
    files = sorted(files, key=lambda x: int(x.split('_')[-1]))
    # delete the first and last file
    files.pop(0)
    files.pop(-1)
    trajectories = {}
    max = -1
    for file in files:
        absolute_path_ = os.path.join(absolute_path, file, 'trajectories.json')
        with open(absolute_path_, 'r') as f:
            data = json.load(f)
            #print(file, data)
            trajectory = np.array(data['positions'])
            # remove the adjacent position duplicates
            trajectory = trajectory[~(np.roll(trajectory, 1, axis=0) == trajectory).all(axis=1)]
            trajectories[file] = trajectory
            max = len(trajectory) if len(trajectory) > max else max
    for key, trajectory in trajectories.items():
        # pad the trajectories with zeros to make them of the same length
        trajectories[key] = np.pad(trajectory, ((0, max - len(trajectory)), (0, 0)))
    # conver the values of the dictionary to tensors
    trajectories_tensor = torch.tensor([trajectory for trajectory in trajectories.values()])
    return trajectories_tensor


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


def visualize_trajectory(trajectory):
    """Visualize the trajectory with points colored in the rainbow spectrum."""
    if isinstance(trajectory, torch.Tensor):
        trajectory = trajectory.numpy()
    elif isinstance(trajectory, np.ndarray):
        pass
    trajectory = trajectory.squeeze()
    x = [point[0] for point in trajectory]
    y = [point[1] for point in trajectory]
    
    # Normalize the colors based on the index of each point in the trajectory
    colors = np.linspace(0, 1, len(x))
    
    plt.figure()
    plt.scatter(x, y, c=colors, cmap='rainbow')
    
    # Invert y-axis to have origin at top-left, and set the axes scales
    plt.gca().invert_yaxis()
    
    # Set the scale of the axes in increments of 1, starting with 0
    x_ticks = np.arange(int(min(x)), int(max(x)) + 2, 1)
    y_ticks = np.arange(int(min(y)), int(max(y)) + 2, 1)
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)
    
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()


def main():
    trajectories = get_trajectories()
    visualize_trajectories(trajectories)

if __name__ == "__main__":
    main()