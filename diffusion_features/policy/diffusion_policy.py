
import math
import os
from inspect import isfunction
from functools import partial

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from einops import rearrange

import torch
from torch import nn, einsum
import torch.nn.functional as F

import numpy as np
from diffusion_features.utils.schedulers import *
from diffusion_features.algo.diffusion_utils import ConditionalUnet1D
from torch.optim import Adam
from diffusion_features.utils.get_trajectories import get_trajectories

from torchvision.utils import save_image
from torch.utils.data import DataLoader
import hydra


class Diffusion_policy():
    def __init__(self, timesteps, scheduler) -> None:
        self.timesteps = timesteps
        self.scheduler = scheduler
        self.calculate_params()
        
    def extract(self, a, t, x_shape)->torch.Tensor:
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    # forward diffusion
    def q_sample(self, x_start, t, noise=None)->torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)

        self.sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return self.sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def calculate_params(self)->None:
        # define beta schedule
        self.betas = self.scheduler(self.timesteps)

        # define alphas
        self.alphas = 1. - self.betas
        alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)


    def get_noisy_image(self, x_start, t)->torch.Tensor:
        # add noise
        x_noisy = self.q_sample(x_start, t=t)

        # turn back into PIL image
        # noisy_image = reverse_transform(x_noisy.squeeze())

        return x_noisy

    def p_losses(self, denoise_model, x_start, t, noise=None, loss_type="l1", condition=None)->torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
            
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        # print("x_noisy", x_noisy.shape)
        predicted_noise = denoise_model(x_noisy, t, global_cond=condition)

        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()

        return loss

    @torch.no_grad()
    def p_sample(self, model, x, t, t_index, global_cond=None)->torch.Tensor:
        betas_t = self.extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = self.extract(self.sqrt_recip_alphas, t, x.shape)

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t, global_cond) / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    # Algorithm 2 but save all images:
    @torch.no_grad()
    def p_sample_loop(self, model, shape, global_cond=None)->list:
        device = next(model.parameters()).device

        b = shape[0]
        # start from pure noise (for each example in the batch)
        img = torch.randn(shape, device=device)
        imgs = []

        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            img = self.p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i, global_cond=global_cond)
            imgs.append(img.cpu().numpy())
        return imgs

    @torch.no_grad()
    def sample(self, model, image_size, batch_size=16, channels=3):
        return self.p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))

    @torch.no_grad()
    def sample_trajectories(self, model, traj_size, batch_size=16, output_dim = 2, global_cond=None,):
        return self.p_sample_loop(model, shape=(batch_size, traj_size, output_dim), global_cond=global_cond)


    def num_to_groups(self, num, divisor)->list:
        groups = num // divisor
        remainder = num % divisor
        arr = [divisor] * groups
        if remainder > 0:
            arr.append(remainder)
        return arr


    def train(self, model, optimizer, epochs, batch_size, trajectories)->None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        # channels = 1
        dataloader = DataLoader(trajectories, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            loss_arr = []
            for step, batch in enumerate(dataloader):
                optimizer.zero_grad()
                
                
                b = batch.shape[0]
                batch = batch.to(device)
                # batch shape: (B, T, C)
                # print('input shape pre reshape', batch.shape)

                # batch = batch.reshape(b, channels, batch.shape[1], batch.shape[2])
                
                # normalize the batch to [-1, 1]
                batch = (batch / 8) * 2 - 1
                
                # generate conditioning based on start and end states
                start_states = batch[:, 0, :]
                end_states = batch[:, -1, :]
                condition = torch.cat((start_states, end_states), axis=1)
                # print(condition)
                condition = None
                # print('condition shape', condition.shape)
                
                # print("batch: ", torch.mean(batch[4]), torch.std(batch[4]))
                # print("Max and min", torch.max(batch[4]), torch.min(batch[4]))

                # Algorithm 1 line 3: sample t uniformally for every example in the batch
                t = torch.randint(0, self.timesteps, (batch_size,), device=device).long()
                if b != batch_size:
                    continue
                loss = self.p_losses(model, batch, t, loss_type="huber", condition=condition)

                # if step % 100 == 0:
                # print("Loss:", loss.item())
                
                loss_arr.append(loss.item())

                loss.backward()
                optimizer.step()
            print(f"Loss for epoch {epoch}: {np.mean(loss_arr)}")
            # save generated images
            # if step != 0 and step % save_and_sample_every == 0:
            #   milestone = step // save_and_sample_every
            #   batches = num_to_groups(4, batch_size)
            #   all_images_list = list(map(lambda n: sample(model, batch_size=n, channels=channels), batches))
            #   all_images = torch.cat(all_images_list, dim=0)
            #   all_images = (all_images + 1) * 0.5
            #   save_image(all_images, str(results_folder / f'sample-{milestone}.png'), nrow = 6)
    
    def sample(self, num_samples, trajectory_length, model, \
                            output_dim, save_path, global_cond=None) -> None:
        save_path = os.path.abspath(save_path)
        # check if the range of global_cond is within -1 and 1
        if global_cond is not None:
            if global_cond.max() > 1 or global_cond.min() < -1:
                global_cond = (global_cond / 8) * 2 - 1
        samples = self.sample_trajectories(model, trajectory_length, batch_size=num_samples,\
                                            output_dim=output_dim, global_cond=global_cond)
        print(len(samples))
        print(samples[0].shape)
        sample_ = samples[self.timesteps-1]
        for j in range(sample_.shape[0]):
            sample_[j] = ((sample_[j] + 1) / 2) * 8
            x = [point[0] for point in sample_[j]]
            y = [point[1] for point in sample_[j]]
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
            figure_path = os.path.join(save_path, f'trajectory_{self.timesteps}_{trajectory_length}_{j}.png')
            plt.savefig(figure_path)


        return samples


@hydra.main(config_path="../conf", config_name="minigrid_conf")
def main(cfg):
    print(cfg)
    epochs = cfg.params.epoch
    batch_size = cfg.params.batch_size
    lr = cfg.params.lr
    timesteps = cfg.params.timesteps
    if cfg.model_params.scheduler == "linear":
        scheduler = linear_beta_schedule
    elif cfg.model_params.scheduler == "cosine":
        scheduler = cosine_beta_schedule
    elif cfg.model_params.scheduler == "quadratic":
        scheduler = quadratic_beta_schedule
    elif cfg.model_params.scheduler == "sigmoid":
        scheduler = sigmoid_beta_schedule
    else:
        raise NotImplementedError("Scheduler not implemented")
    # trajectories = trajectories[0:32]
    trajectories = get_trajectories(cfg.paths.data_path)
    trajectories = torch.tensor(trajectories, dtype=torch.float32)
    # x_start = trajectories[30].unsqueeze(0)
    # x_start = x_start.float()
    # print(x_start)

    trajectories = trajectories[:cfg.params.num_trajectories]
    # reduce the trajectory size to 16
    trajectories = trajectories[:, :cfg.params.trajectory_len, :]
    trajectories_size = trajectories.shape
    # model = Unet(
    #     dim=trajectories_size[1],
    #     channels=channels,
    #     dim_mults=(1, 2, 4,)
    # )
    # TODO: double check down_dims
    model = ConditionalUnet1D(
        input_dim=cfg.model_params.input_dim,
        global_cond_dim=cfg.model_params.global_cond_dim,
        # global_cond_dim=4,
        down_dims=cfg.model_params.down_dims,
        diffusion_step_embed_dim=trajectories_size[1],
        kernel_size=cfg.model_params.kernel_size,
        n_groups=cfg.model_params.n_groups,
    )

    optimizer = Adam(model.parameters(), lr=lr)
    policy = Diffusion_policy(timesteps, scheduler=scheduler)
    policy.train(model, optimizer, epochs, batch_size, trajectories)
    policy.sample(cfg.params.num_samples, cfg.params.trajectory_len,\
                                model, cfg.model_params.output_dim, cfg.paths.save_path)

if __name__=="__main__":
    main()