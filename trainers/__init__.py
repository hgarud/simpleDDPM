"""Trainer registry."""

from .ddpm import DDPMTrainer


ALL_TRAINERS = {
    'ddpm': DDPMTrainer,
}


def get_trainer(name: str, num_time_steps: int):
    """Return the trainer corresponding to the given name."""
    if name not in ALL_TRAINERS:
        raise ValueError(f'Unknown trainer: {name}')
    return ALL_TRAINERS[name](num_time_steps)
