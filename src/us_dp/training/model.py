"""Multimodal encoder + upstream conditional U-Net + DDPM epsilon objective."""

import math

import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch import nn
from torch.nn import functional as F

from us_dp.common.spline import SplineCodec
from us_dp.common.upstream import classes
from us_dp.training.usfm_encoder import USFMEncoder


class UltrasoundSplinePolicy(nn.Module):
    def __init__(self, config, state_dim, repo=None, *, load_pretrained=True):
        super().__init__()
        if state_dim not in (9, 23):
            raise ValueError("The policy requires a 9-value Cartesian pose or full 23-value robot state")
        self.config = config
        self.state_dim = state_dim
        _, unet = classes(repo)
        self.codec = SplineCodec(config.num_segments, config.future_steps + 1, repo)
        f = config.feature_dim
        self.image_encoder = (USFMEncoder(
            config, pretrained=config.usfm_pretrained if load_pretrained else None, freeze=config.usfm_freeze
        ) if config.use_image_conditioning else None)
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
            # Cosine epsilon reconstruction divides by sqrt(alpha_bar), which
            # is almost zero at the first reverse step. Bound x0 before it enters
            # the posterior, not just the final trajectory after divergence.
            clip_sample=config.sampling_clip_range is not None,
            clip_sample_range=config.sampling_clip_range or 1.0,
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
        state_features = self.state_encoder(((state - self.state_mean) / self.state_std).flatten(1))
        image_features = (
            self.image_encoder(image.squeeze(2))
            if self.image_encoder is not None else torch.zeros_like(state_features)
        )
        return torch.cat((image_features, state_features), dim=-1)

    def predict_noise(self, noisy, timestep, condition):
        padded = F.pad(noisy, (0, 0, 0, self.padded_count - self.free_count))
        return self.denoiser(padded, timestep, global_cond=condition)[:, : self.free_count]

    def normalize_params(self, free):
        return (free - self.param_mean) / self.param_std

    def denormalize_params(self, normalized):
        return normalized * self.param_std + self.param_mean

    def diffusion_batch(self, batch):
        """Expose the actual DDPM training tensors for diagnostics."""
        cond = self.condition(batch["ultrasound"], batch["robot_state"])
        clean = self.normalize_params(batch["spline_params"][:, 1:])
        noise = torch.randn_like(clean)
        t = torch.randint(self.config.diffusion_steps, (len(clean),), device=clean.device)
        noisy = self.scheduler.add_noise(clean, noise, t)
        predicted = self.predict_noise(noisy, t, cond)
        return clean, noise, noisy, predicted, cond

    def compute_loss(self, batch):
        _, noise, _, predicted, _ = self.diffusion_batch(batch)
        return F.mse_loss(predicted, noise)

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
        free = self.denormalize_params(latent)
        params = self.codec.unpack(free.flatten(1))
        return {"spline_params": params, "trajectory": self.codec.decode(params)}
