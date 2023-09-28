from typing import Optional

import torch


def rejection_sample(
    samples: torch.Tensor, log_w: torch.Tensor
) -> torch.Tensor:
    """Perform rejection sampling given samples and log-weights."""
    log_u = torch.log(torch.rand_like(log_w))
    log_w = log_w - log_w.max()
    return samples[log_w > log_u]


def sample_nball(
    dims: int,
    n: int = 1,
    radius: float = 1.0,
    device: Optional[torch.DeviceObjType] = None,
) -> torch.Tensor:
    """Sample from an n-ball with radius r"""
    z = torch.randn(n, dims, device=device)
    z = z / torch.sqrt(torch.sum(z**2, dim=1, keepdim=True))
    r = radius * (torch.rand(n, 1, device=device) ** (1 / dims))
    return r * z
