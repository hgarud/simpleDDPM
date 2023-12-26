"""Trainer registry."""

import torch

from .ddpm import DDPMTrainer


ALL_TRAINERS = {
    'ddpm': DDPMTrainer,
}


def get_trainer(name: str, denoiser: torch.nn.Module, num_time_steps: int):
    """Return the trainer corresponding to the given name."""
    if name not in ALL_TRAINERS:
        raise ValueError(f'Unknown trainer: {name}')
    return ALL_TRAINERS[name](denoiser, num_time_steps)
