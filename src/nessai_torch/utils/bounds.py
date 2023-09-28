"""Utilities for handling bounds"""
import torch


@torch.jit.script
def in_bounds(x: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
    return ~(
        torch.any(x < bounds[0], dim=-1) | torch.any(x > bounds[1], dim=-1)
    )
