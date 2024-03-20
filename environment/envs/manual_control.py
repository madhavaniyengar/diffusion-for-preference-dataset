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
import random

def check_folder_existence(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

# Function to save the environment's image
def save_environment_image(env, save_path):
    check_folder_existence(save_path)
    image_array = env.get_frame(highlight=False, agent_pov=False)
    pygame_image = pygame.surfarray.make_surface(image_array.transpose(1, 0, 2))
    pygame.image.save(pygame_image, f"{save_path}/start_state_image.png")

def save_observations(observations: dict, save_path: str):
    """Save observations and optionally an environment start state image, then return new save path."""
    print('in save observations')
    if save_path:
        print(f'save path {save_path} exists')
        check_folder_existence(save_path)
        with open(f"{save_path}/trajectories.json", 'w') as f:
            print(f'writing to file {save_path}/trajectories.json')
            json.dump(observations, f, default=lambda x: x.tolist() if hasattr(x, "tolist") else x.__dict__)
    
    new_index = len(os.listdir("diffusion_features/data/lavaenv"))
    print(f'returning new save path with index {new_index}')
    return {"positions": [], "actions": [], "reward": [], "mission": ""}, f"diffusion_features/data/lavaenv/manual_control_{new_index}"

class ManualControl(ManualControl):
    def __init__(self, env: MiniGridEnv, seed: int) -> None:
        super().__init__(env, seed)
        self.observations, self.save_path = save_observations(None, None)  # Pass the environment for initial image save
        save_environment_image(self.env, self.save_path)

    def step(self, action: Actions):
        obs, reward, terminated, truncated, _ = self.env.step(action)
        position = self.env.agent_pos
        self.observations['positions'].append(position)
        self.observations['actions'].append(action)
        self.observations['reward'].append(reward)
        print(f"step={self.env.step_count}, reward={reward:.2f}")
        if terminated or truncated:
            print("terminated!" if terminated else "truncated!")
            self.observations['mission'] = obs['mission']
            self.observations, self.save_path = save_observations(self.observations, self.save_path)  # Image save not needed here
            self.env.reset()
            save_environment_image(self.env, self.save_path)
        else:
            self.env.render()

# Note: Ensure pygame is initialized before saving images
pygame.init()
