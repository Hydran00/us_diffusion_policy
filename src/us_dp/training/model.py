"""Multimodal encoder + upstream conditional U-Net + DDPM epsilon objective."""

import math

import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch import nn
from torch.nn import functional as F

from us_dp.common.spline import SplineCodec
from us_dp.common.upstream import classes


class UltrasoundSplinePolicy(nn.Module):
    def __init__(self, config, state_dim, repo=None):
        super().__init__()
        self.config = config
        self.state_dim = state_dim
        _, unet = classes(repo)
        self.codec = SplineCodec(config.num_segments, config.future_steps + 1, repo)
        f = config.feature_dim
        # Stack ultrasound history as channels; all frames precede/current anchor.
        self.image_encoder = nn.Sequential(
            nn.Conv2d(config.history, 32, 5, stride=2, padding=2),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 16, f),
            nn.SiLU(),
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim * config.history, 256),
            nn.SiLU(),
            nn.Linear(256, f),
            nn.SiLU(),
        )
        self.denoiser = unet(
            input_dim=3,
            global_cond_dim=2 * f,
            diffusion_step_embed_dim=128,
            down_dims=config.down_dims,
            kernel_size=3,
            n_groups=8,
            cond_predict_scale=True,
        )
        # K+1 free variables after fixing the initial position. Pad only inside
        # the denoiser for the upstream U-Net's down/up-sampling divisibility.
        self.free_count = config.num_segments + 1
        factor = 2 ** (len(config.down_dims) - 1)
        self.padded_count = math.ceil(self.free_count / factor) * factor
        self.scheduler = DDPMScheduler(
            num_train_timesteps=config.diffusion_steps,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon",
            clip_sample=False,
        )
        self.register_buffer("state_mean", torch.zeros(state_dim))
        self.register_buffer("state_std", torch.ones(state_dim))
        self.register_buffer("param_mean", torch.zeros(3))
        self.register_buffer("param_std", torch.ones(3))

    def set_statistics(self, statistics):
        for prefix, key in [("state", "robot_state"), ("param", "spline_params")]:
            mean, std = statistics[key]
            getattr(self, prefix + "_mean").copy_(mean)
            getattr(self, prefix + "_std").copy_(std)

    def condition(self, image, state):
        c = self.config
        if image.shape[1:] != (c.history, 1, c.image_size, c.image_size) or state.shape[1:] != (
            c.history,
            self.state_dim,
        ):
            raise ValueError("Observation shape differs from the training configuration")
        if (
            not torch.isfinite(image).all()
            or not torch.isfinite(state).all()
            or image.min() < 0
            or image.max() > 1
        ):
            raise ValueError("Finite observations and ultrasound in [0,1] required")
        return torch.cat(
            (
                self.image_encoder(image.squeeze(2)),
                self.state_encoder(((state - self.state_mean) / self.state_std).flatten(1)),
            ),
            dim=-1,
        )

    def predict_noise(self, noisy, timestep, condition):
        padded = F.pad(noisy, (0, 0, 0, self.padded_count - self.free_count))
        return self.denoiser(padded, timestep, global_cond=condition)[:, : self.free_count]

    def compute_loss(self, batch):
        cond = self.condition(batch["ultrasound"], batch["robot_state"])
        clean = (batch["spline_params"][:, 1:] - self.param_mean) / self.param_std
        noise = torch.randn_like(clean)
        t = torch.randint(self.config.diffusion_steps, (len(clean),), device=clean.device)
        noisy = self.scheduler.add_noise(clean, noise, t)
        return F.mse_loss(self.predict_noise(noisy, t, cond), noise)

    @torch.no_grad()
    def predict(self, image, state, generator=None):
        cond = self.condition(image, state)
        latent = torch.randn(
            (len(image), self.free_count, 3), device=image.device, generator=generator
        )
        self.scheduler.set_timesteps(self.config.inference_steps, device=image.device)
        for t in self.scheduler.timesteps:
            noise = self.predict_noise(latent, t, cond)
            latent = self.scheduler.step(noise, t, latent, generator=generator).prev_sample
        free = latent * self.param_std + self.param_mean
        params = torch.cat((torch.zeros_like(free[:, :1]), free), dim=1)
        return {"spline_params": params, "trajectory": self.codec.decode(params)}
