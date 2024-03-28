import torch
import numpy as np
import random
import os
import json
from collections import deque
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/sagarpatil/sagar/projects/diffusion-features/')
from environment.envs.lava import LavaEnv



# generate trajectories for the lava environment using A* algorithm with minimum length of trajectory as 8
def generate_trajectories():
    env = LavaEnv()
    env.reset()
    # generate 1000 trajectories
    for i in range(1000):
        observations = {}
        # generate a random goal position
        goal = (np.random.randint(1, env.size-1), np.random.randint(1, env.size-1))
        # generate a random start position
        start = (np.random.randint(1, env.size-1), np.random.randint(1, env.size-1))
        # check if the distance between the start and goal is greater than 8
        while env.distance(start, goal) < 8:
            start = (np.random.randint(1, env.size-1), np.random.randint(1, env.size-1))
        # get the trajectory
        trajectory = env.astar_path(start, goal)
         # convert the trajectory to a list of tuples
        trajectory = [[point[0], point[1]] for point in trajectory]
        observations['positions'] = trajectory
        # create a folder for each trajectory
        os.makedirs(f'../../environment/data/lavaenv/manual_control_{i+35}', exist_ok=True)
        with open(f'../../environment/data/lavaenv/manual_control_{i+35}/trajectories.json', 'w') as f:
            json.dump(observations, f, default=lambda x: x.tolist() if hasattr(x, "tolist") else x.__dict__)

if __name__ == '__main__':
    generate_trajectories()