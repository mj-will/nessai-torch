"""Proposal that samples from the prior subject to any constraints"""
import torch
from typing import Callable, Tuple

from .base import Proposal


class PriorProposal(Proposal):
    """Proposal that samples from the prior subject to any constraints.

    Evaluates the likelihood of samples
    """

    def draw(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        """Draw a valid sample from the prior"""
        logl = torch.tensor(-torch.inf)
        while not torch.isfinite(logl):
            x = torch.rand(1, self.dims, device=self.device)
            logl = self.log_likelihood_fn(x)
        return x, logl
