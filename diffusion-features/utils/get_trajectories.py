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
    files = os.listdir('diffusion-features/data/lavaenv')
    trajectories = []
    for file in files:
        with open(f'diffusion-features/data/lavaenv/{file}', 'r') as f:
            data = json.load(f)
            trajectories.append(data["positions"])


def visualize_trajectories(trajectories):
    """Visualize the trajectories."""
    for trajectory in trajectories:
        x = [point[0] for point in trajectory]
        y = [point[1] for point in trajectory]
        plt.plot(x, y)
    plt.show()
