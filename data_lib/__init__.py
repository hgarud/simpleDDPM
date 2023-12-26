"""Dataset registry."""
import torch

from . import cifar10


ALL_DATASETS = {
    "cifar10": cifar10.get_dataset,
}


def get_dataset(name: str, root: str, train: bool) -> torch.utils.data.Dataset:
    """Return the dataset corresponding to the given name."""
    if name not in ALL_DATASETS:
        raise ValueError(f"Unknown dataset: {name}")
    return ALL_DATASETS[name](root, train)