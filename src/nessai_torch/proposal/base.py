from abc import abstractmethod
from typing import Callable, List

import torch


class Proposal:
    has_pool: bool = False
    populated: bool = False
    trainable: bool = False

    def __init__(self, *, dims: int, device: torch.DeviceObjType) -> None:
        self.dims = dims
        self.device = device

    @abstractmethod
    def draw(self) -> None:
        raise NotImplementedError


class ProposalWithPool(Proposal):
    has_pool: bool = True
    populated: bool = False
    trainable: bool = False
    indices: List[int] = None
    samples: torch.Tensor = None
    logl: torch.Tensor = None

    @abstractmethod
    def populate(self, live_points: torch.Tensor, logl: torch.Tensor) -> None:
        raise NotImplementedError

    def compute_likelihoods(self, log_likelihood_fn: Callable) -> None:
        if self.samples is None or not self.populated:
            raise RuntimeError("Proposal is not populated")
        self.logl = log_likelihood_fn(self.samples)

    def draw(self, x: torch.Tensor) -> torch.Tensor:
        if not self.populated:
            raise RuntimeError(
                "Proposal must be populated before drawing a sample"
            )
        index = self.indices.pop()
        new_sample = self.samples[index]
        if not self.indices:
            self.populated = False
            self.samples = None
        if self.logl is None:
            new_logl = None
        else:
            new_logl = self.logl[index]
        return new_sample, new_logl
