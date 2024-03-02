import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
import json
from collections import deque
import matplotlib.pyplot as plt
from typing import Tuple
from torch import Tensor
from torch.optim import Adam #the optimizer
from torch.distributions import Categorical #the distribution


# TODO: 
# State Sequence Encoder -> Encodes the state



# ConditionUnet1d
# INPUT: random_actions, states, timestep
# OUTPUT: predicted noise

# Noise Scheduler
# 
class DiffusionPolicy(Policy):
    def __init__(
        self,
        state_sequence_encoder: StateSequenceEncoder,
        noise_pred_net: ConditionalUnet1D,
        device: torch.device,
        dtype: torch.dtype,
        action_dim: int,
        noise_scheduler: NoiseScheduler,
        action_norm_config: NormalizationConfig,
        obs_horizon: int,
        action_horizon: int,
        seed: int,
        **kwargs,
    ):
        Policy.__init__(self, supported_tasks=supported_tasks)
        self.state_sequence_encoder = state_sequence_encoder.eval()
        self.noise_pred_net = noise_pred_net.eval()
        self.device = device
        self.dtype = dtype
        self.noise_scheduler = noise_scheduler
        self.action_dim = action_dim
        self.action_norm_config = action_norm_config
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
        B = actions.size[0]
        normalized_random_actions = torch.randn((B, self.actions.shape[1], self.actions.shape[2]), 
                                                device=self.device, dtype=self.dtype)
        for timestep in self.noise_scheduler.timesteps:
            noise_pred = self.noise_pred_net(normalized_random_actions, states, timestep)
            normalized_random_actions = self.noise_scheduler.step(
                        model_output=noise_pred,
                        timestep=int(timestep),
                        sample=normalized_random_actions,
                        generator=self.get_generator()
                    ).prev_sample
            return self.unnormalize(normalized_random_actions)
        
                    


