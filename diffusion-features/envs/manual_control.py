import os
import json
import gymnasium as gym
import pygame
from gymnasium import Env

from minigrid.core.actions import Actions
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
from minigrid.manual_control import ManualControl
from minigrid.wrappers import FullyObsWrapper


def save_observations(observations: dict, save_path: str):
    """Save observations and return new save path based on the number of files in the directory."""
    if save_path is not None:
        with open(save_path, 'w') as f:
            # Ensure numpy arrays or custom objects are converted to lists or dicts
            json.dump(observations, f, default=lambda x: x.tolist() if hasattr(x, "tolist") else x.__dict__)
    return {"states": [], "mission": "", "truncated": False}, f"diffusion-features/data/lavaenv/manual_control_{len(os.listdir('diffusion-features/data/lavaenv'))}.json"

class ManualControl(ManualControl):
    def __init__(self, env: MiniGridEnv, seed: int) -> None:
        super().__init__(env, seed)
        
        self.observations, self.save_path = save_observations(None, None)

    def step(self, action: Actions):
        obs, reward, terminated, truncated, _ = self.env.step(action)
        print(f"step={self.env.step_count}, reward={reward:.2f}")
        if terminated:
            print("terminated!")
            self.observations['mission'] = obs['mission']
            self.observations['truncated'] = False
            self.observations, self.save_path = save_observations(self.observations, self.save_path)
            self.reset(self.seed)

        elif truncated:
            print("truncated!")
            self.observations['mission'] = obs['mission']
            self.observations['truncated'] = True
            self.observations, self.save_path = save_observations(self.observations, self.save_path)
            self.reset(self.seed)
        else:
            # Ensure the observation's image and direction are stored as lists (or appropriate structures)
            self.observations['states'].append({
                "image": obs['image'].tolist() if hasattr(obs['image'], 'tolist') else obs['image'],
                "direction": obs['direction'].tolist() if hasattr(obs['direction'], 'tolist') else obs['direction'],
                "action": action,
                "reward": reward
            })
            self.env.render()