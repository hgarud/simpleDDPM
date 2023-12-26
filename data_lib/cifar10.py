"""Cifar10 dataset module."""

import torch
import torchvision
import torchvision.transforms as transforms

def get_dataset(root: str, batch_size: int, train: bool) -> torch.utils.data.Dataset:
    # Define the transformation to apply to the dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load the CIFAR-10 training dataset
    dataset = torchvision.datasets.CIFAR10(root=root,
                                            train=train,
                                            transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=(train==True), num_workers=2)

    return dataloader
