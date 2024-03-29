import numpy as np
import random
import os
import json
from collections import deque
import matplotlib.pyplot as plt
import torch

def pad_trajectories(trajectories, max_length):
    current_length = len(trajectories)
    
    # Check if padding is needed
    if current_length >= max_length:
        return trajectories[:max_length]
    
    pad_size = max_length - current_length
    begin_pad_size = pad_size // 2
    end_pad_size = pad_size - begin_pad_size
    
    begin_pad = [list(trajectories[0])] * begin_pad_size
    end_pad = [list(trajectories[-1])] * end_pad_size
    
    padded_trajectories = begin_pad + list(trajectories) + end_pad
    
    return padded_trajectories


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
    # if max is odd, make it even
    max = max + 1 if max % 2 != 0 else max
    # get the nearest power of 2
    max = 2**np.ceil(np.log2(max)).astype(int)
    for key, trajectory in trajectories.items():
        # pad the trajectories with zeros to make them of the same length
        # trajectories[key] = np.pad(trajectory, ((0, max - len(trajectory)), (0, 0)))
        # pad the trajectories with the last point to make them of the same length
        # trajectories[key] = np.pad(trajectory, ((0, max - len(trajectory)), (0, 0)), 'edge')
        trajectories[key] = pad_trajectories(trajectory, max)
    # conver the values of the dictionary to tensors
    trajectories_tensor = torch.tensor([trajectory for trajectory in trajectories.values()])
    print(trajectories_tensor.shape)
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

def discretize_trajectories():
    # discretize the first 35 trajectories
    absolute_path = '/Users/sagarpatil/sagar/projects/diffusion-features/environment/data/lavaenv'
    files = os.listdir(absolute_path)
    # sort the files according to name
    files = sorted(files, key=lambda x: int(x.split('_')[-1]))
    for file in files[:35]:
        absolute_path_ = os.path.join(absolute_path, file, 'trajectories.json')
        with open(absolute_path_, 'r') as f:
            data = json.load(f)
            trajectory = np.array(data['positions'])
            # remove the adjacent position duplicates
            trajectory = trajectory[~(np.roll(trajectory, 1, axis=0) == trajectory).all(axis=1)]
            new_trajectory = []
            for k in range(len(trajectory)-1):
                point1 = trajectory[k]
                point2 = trajectory[k+1]
                x1, y1 = point1
                x2, y2 = point2
                x = np.linspace(x1, x2, int(np.ceil(np.linalg.norm(point1 - point2)/0.2)))
                y = np.linspace(y1, y2, int(np.ceil(np.linalg.norm(point1 - point2)/0.2)))
                for j in range(len(x)):
                    new_trajectory.append([x[j], y[j]])
            # remove the adjacent position duplicates
            new_trajectory = np.array(new_trajectory)
            new_trajectory = new_trajectory[~(np.roll(new_trajectory, 1, axis=0) == new_trajectory).all(axis=1)]
            # convert the numpy array to a list
            new_trajectory = new_trajectory.tolist()
            data['positions'] = new_trajectory
            with open(absolute_path_, 'w') as f:
                json.dump(data, f, default=lambda x: x.tolist() if hasattr(x, "tolist") else x.__dict__)

if __name__ == "__main__":
    discretize_trajectories()