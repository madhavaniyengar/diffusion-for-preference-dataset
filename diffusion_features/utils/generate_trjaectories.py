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
        # discretize the trajectory to a finer resolution of 0.25 units by interpolating between the points
        new_trajectory = []
        for k in range(len(trajectory)-1):
            point1 = trajectory[k]
            point2 = trajectory[k+1]
            x1, y1 = point1
            x2, y2 = point2
            x = np.linspace(x1, x2, int(np.ceil(env.distance(point1, point2)/0.2)))
            y = np.linspace(y1, y2, int(np.ceil(env.distance(point1, point2)/0.2)))
            # print(x, y)
            for j in range(len(x)):
                new_trajectory.append([x[j], y[j]])
         # convert the trajectory to a list of tuples
        # trajectory = [[point[0], point[1]] for point in trajectory]
        # convert adjacency duplicates to a single point
        new_trajectory = np.array(new_trajectory)
        new_trajectory = new_trajectory[~(np.roll(new_trajectory, 1, axis=0) == new_trajectory).all(axis=1)]
        # convert numpy array to list
        new_trajectory = new_trajectory.tolist()
        observations['positions'] = new_trajectory
        # create a folder for each trajectory
        os.makedirs(f'../../environment/data/lavaenv/manual_control_{i+35}', exist_ok=True)
        with open(f'../../environment/data/lavaenv/manual_control_{i+35}/trajectories.json', 'w') as f:
            json.dump(observations, f, default=lambda x: x.tolist() if hasattr(x, "tolist") else x.__dict__)

if __name__ == '__main__':
    generate_trajectories()