from abc import abstractmethod
from typing import Callable, List, Optional

import torch


class Proposal:
    """ "Base class for all proposals."""

    has_pool: bool = False
    """Indicates if the proposal has a pool of samples."""
    trainable: bool = False
    """Indicates if the proposal can be trained."""

    def __init__(
        self,
        *,
        dims: int,
        device: torch.DeviceObjType,
        log_likelihood_fn: Callable,
    ) -> None:
        """
        Parameters
        ----------
        dims
            Dimensionality of the parameter space.
        device
            Torch device.
        log_likelihood_fn
            Function for evaluating the log-likelihood for samples in the
            unit hypercube space.
        """
        self.dims = dims
        self.device = device
        self.log_likelihood_fn = log_likelihood_fn

    @abstractmethod
    def draw(self, x: torch.Tensor) -> None:
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

    def compute_likelihoods(
        self, log_likelihood_fn: Optional[Callable] = None
    ) -> None:
        """Compute the log-likelihoods for the pool of samples.

        Parameters
        ----------
        log_likelihood_fn
            Function to use instead of the function that was specified when
            instantiating the class.
        """
        if self.samples is None or not self.populated:
            raise RuntimeError("Proposal is not populated")
        if log_likelihood_fn is None:
            log_likelihood_fn = self.log_likelihood_fn
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
