from typing import Optional, Tuple

import torch


@torch.jit.script
def logit_with_jacobian(
    x: torch.Tensor, eps: Optional[float] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    if eps is not None:
        x = torch.clamp(x, eps, 1 - eps)
    log_j = -torch.log(x) - torch.log1p(-x)
    return torch.logit(x), log_j


@torch.jit.script
def sigmoid_with_jacobian(
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    x = torch.sigmoid(x)
    log_j = torch.log(x) + torch.log1p(-x)
    return x, log_j
