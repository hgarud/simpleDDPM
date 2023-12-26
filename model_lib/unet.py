""" The denoising UNet model. """

import math
import torch.nn as nn


# Define model
class UNet(nn.Module):
    def __init__(self, t_embed_dim: int):
        self.t_embed_dim = t_embed_dim
        

    def get_time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """
        Returns the time embedding for the given time tensor.
        Args:
            t: Input tensor of shape (batch_size, 1)
        Returns:
            Time embedding of shape (batch_size, embedding_dim)
        """
        half_dim = self.t_embed_dim // 2
        emb = math.log(10000.0) / (half_dim - 1)
        emb = torch.arange(half_dim, dtype=torch.float32) * -emb
        emb = torch.exp(emb)
        emb = t * emb.unsqueeze(0)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_embedding = self.get_time_embedding(t)
