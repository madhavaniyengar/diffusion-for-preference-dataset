import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import torch
import torch.nn as nn
import torch.nn.functional as F
from algo.diffusion_utils import ConditionalUnet1D
import numpy as np
import random
import os
import json
from collections import deque
import matplotlib.pyplot as plt
from typing import Tuple, Union
from torch import Tensor
from torch.optim import Adam #the optimizer
from torch.distributions import Categorical #the distribution
from diffusers import DDPMScheduler, DDIMScheduler
NoiseScheduler = Union[DDPMScheduler, DDIMScheduler]



# TODO: 
# State Sequence Encoder -> Encodes the state



# ConditionUnet1d
# INPUT: random_actions, states, timestep
# OUTPUT: predicted noise

# Noise Scheduler
# 
class DiffusionPolicy():
    def __init__(
        self,
        # state_sequence_encoder: StateSequenceEncoder,
        noise_pred_net: ConditionalUnet1D,
        device: torch.device,
        dtype: torch.dtype,
        action_dim: int,
        noise_scheduler: NoiseScheduler,
        # action_norm_config: NormalizationConfig,
        obs_horizon: int,
        action_horizon: int,
        seed: int,
        **kwargs,
    ):
        # Policy.__init__(self, supported_tasks=supported_tasks)
        # self.state_sequence_encoder = state_sequence_encoder.eval()
        self.noise_pred_net = noise_pred_net.eval()
        self.device = device
        self.dtype = dtype
        self.noise_scheduler = noise_scheduler
        self.action_dim = action_dim
        # self.action_norm_config = action_norm_config
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.seed = seed
        self.noise_pred_net = noise_pred_net.to(device)

    def get_generator(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        return generator
    
    

    def train(self, states: Tensor, actions: Tensor): 
        """
        Train the noise_pred_net based on the states as conditional input 
        and actions as targets
        states: Tensor of shape (B, obs_horizon, state_dim)
        actions: Tensor of shape (B, action_horizon, action_dim)
        """
        states = states.to(self.device)
        actions = actions.to(self.device)
        B = actions.shape[0]
        normalized_random_actions = torch.randn((B, actions.shape[1], actions.shape[2]),
                                                device=self.device, dtype=self.dtype)
        noise = torch.rand_like(actions, device=self.device, dtype=self.dtype)
        noisy_action = self.noise_scheduler.add_noise(actions, noise, timesteps=torch.arange(0, 1000))
        print(noisy_action)
        for timestep in self.noise_scheduler.timesteps:
            noise_pred = self.noise_pred_net(states, timestep)
            normalized_random_actions = self.noise_scheduler.step(
                        model_output=noise_pred,
                        timestep=int(timestep),
                        sample=normalized_random_actions,
                        generator=self.get_generator()
                    ).prev_sample
            return self.unnormalize(normalized_random_actions)


if __name__ == "__main__":
    # Test the DiffusionPolicy class
    states = torch.tensor([[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]], dtype=torch.float16)
    actions = torch.tensor([[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]], dtype=torch.float16)
    noise_pred_net = ConditionalUnet1D(
        input_dim=5,
        global_cond_dim=1,
        down_dims=[16, 32, 64],
        kernel_size=3,
        n_groups=1,
        diffusion_step_embed_dim=16,
    )
    scheduler = DDPMScheduler()
    diffusion_policy = DiffusionPolicy(
        noise_pred_net=noise_pred_net,
        dtype=torch.float16,
        action_dim=5,
        noise_scheduler=scheduler,
        obs_horizon=2,
        action_horizon=2,
        seed=42,
        device="cpu"
    )
    diffusion_policy.train(states, actions)