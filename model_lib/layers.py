"""Commonly used layers."""

import torch
import torch.nn as nn


class ResNetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float, shortcut_conv: bool = False):
        super().__init__()

        self.first_half = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.LazyLinear(out_channels)
            )
        self.second_half = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        if in_channels != out_channels:
            if shortcut_conv:
                self.skip = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            else:
                self.skip = nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = x
        h = self.first_half(h)
        h = h + self.time_projection(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.second_half(h)
        if hasattr(self, 'skip'):
            x = self.skip(x)
        return x + h
