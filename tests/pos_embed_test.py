import math
import torch

def get_time_embedding(t: torch.Tensor, t_embed_dim: int = 128) -> torch.Tensor:
    """
    Returns the time embedding for the given time tensor.
    Args:
        t: Input tensor of shape (batch_size, 1)
        t_embed_dim: Dimensions of the time embedding
    Returns:
        Time embedding of shape (batch_size, embedding_dim)
    """
    half_dim = t_embed_dim // 2
    emb = math.log(10000.0) / (half_dim - 1)
    emb = torch.arange(half_dim, dtype=torch.float32) * -emb
    print(emb.shape)
    emb = torch.exp(emb)
    emb = t * emb.unsqueeze(0)
    return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

if __name__ == '__main__':
    t = torch.randint(low = 0, high = 100, size = (5, 1))
    emdedding_dim = 128
    t_embed = get_time_embedding(t, t_embed_dim=emdedding_dim)
    print(t_embed.shape[-1] == emdedding_dim)
 