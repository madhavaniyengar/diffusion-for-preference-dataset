import gymnasium as gym
import pygame
from gymnasium import Env

from minigrid.core.actions import Actions
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
from minigrid.manual_control import ManualControl
from minigrid.wrappers import FullyObsWrapper

class ManualControl(ManualControl):
    def __init__(self, env: MiniGridEnv, seed: int, save_path: str = None) -> None:
        super().__init__(env, seed)
        self.save_path = save_path
        self.t = 0
    

    def step(self, action: Actions):
        obs, reward, terminated, truncated, _ = self.env.step(action)
        print(f"step={self.env.step_count}, reward={reward:.2f}")
        if terminated:
            print("terminated!")
            if self.save_path is not None:
                # open the file in append mode
                with open(self.save_path, "a") as file:
                    file.write("Mission: ")
                    file.write(obs['mission'])
                    file.write("\n")
                    file.write(str(reward))
            self.reset(self.seed)

        elif truncated:
            print("truncated!")
            self.reset(self.seed)
        else:
             # write the observation to the save_path
            if self.save_path is not None:
                # open the file in append mode
                with open(self.save_path, "a") as file:
                    file.write(f"Observation at time step {self.t}")
                    file.write(str(obs['image']))
                    file.write("\n")
                    file.write("Direction: ")
                    file.write(str(obs['direction']))
                    file.write("\n")
                    # Add action also
                
         
            self.env.render()
        self.t += 1
