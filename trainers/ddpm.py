"""Denoising Diffusion Probabilistic Models (DDPM) trainer."""

import torch

class DDPMTrainer():
    def __init__(self, num_time_steps: int):
        self.num_time_steps = num_time_steps
        self.betas = self.set_variance_schedule()
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = self.alphas.cumprod(dim=1)
    
    def q_sample(self, x_0: torch.Tensor, t: int) -> torch.Tensor:
        """Sample q(x_t | x_0) using Eq. 4 of https://arxiv.org/pdf/2006.11239.pdf"""
        return torch.sqrt(self.alpha_cumprod[:, t, ...].unsqueeze(1)) * x_0 + torch.sqrt(1.0 - self.alpha_cumprod[:, t, ...].unsqueeze(1)) * torch.randn_like(x_0)

    def set_variance_schedule(self):
        """Set the variance schedule. Ref: Sec. 4 of https://arxiv.org/pdf/2006.11239.pdf"""
        # Linear schedule from 1e-4 to 0.02
        return torch.linspace(1e-4, 0.02, self.num_time_steps).view(1, -1, 1, 1)

    def __call__(self, x_0: torch.Tensor, t: int):
        return self.q_sample(x_0, t)
