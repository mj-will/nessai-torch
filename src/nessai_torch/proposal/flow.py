import copy
import logging
import math
import os
from typing import Callable, Optional, Tuple, Union

import glasflow
import matplotlib.pyplot as plt
from scipy import stats
import torch

from .base import ProposalWithPool

from ..flows import get_realnvp
from ..tensorlist import TensorList
from ..transforms import logit_with_jacobian, sigmoid_with_jacobian
from ..utils.sample import (
    sample_nball,
    sample_radially_truncated_gaussian,
)

logger = logging.getLogger(__name__)


class FlowProposal(ProposalWithPool):
    """Proposal that draws samples using a normalizing flow"""

    trainable: bool = True
    flow: glasflow.flows.base.Flow = None
    scale: torch.Tensor = None
    shift: torch.Tensor = None
    flow_config: dict = None

    def __init__(
        self,
        *,
        dims: int,
        device: torch.DeviceObjType,
        log_likelihood_fn: Callable,
        poolsize: int,
        batch_size: int = 50_000,
        logit: bool = False,
        constant_volume_fraction: Optional[float] = None,
        truncate_log_q: bool = False,
        sample_nball: bool = False,
        max_samples: int = 1_000_000,
        flow_config: Optional[dict] = None,
        accumulate_weights: bool = False,
    ) -> None:
        super().__init__(
            dims=dims, device=device, log_likelihood_fn=log_likelihood_fn
        )

        self.poolsize = poolsize
        self.batch_size = batch_size
        self.truncate_log_q = truncate_log_q
        self.sample_nball = sample_nball
        self.logit = logit
        self.max_samples = max_samples
        self.count = 0
        self.accumulate_weights = accumulate_weights

        self.configure_flow(flow_config)
        self.configure_constant_volume_mode(constant_volume_fraction)

    def configure_constant_volume_mode(
        self, constant_volume_fraction: float
    ) -> None:
        """Configure constant volume mode.

        Sets the radius and u_max (is :code:`sample_nball=False`)
        """
        self.constant_volume_fraction = constant_volume_fraction
        self.u_max = None
        if self.constant_volume_fraction is not None:
            self.cvm_radius = torch.tensor(
                stats.chi.ppf(self.constant_volume_fraction, self.dims)
            )
            if not self.sample_nball:
                self.u_max = torch.tensor(self.constant_volume_fraction)
            logger.debug(f"CVM radius: {self.cvm_radius.item():.3f}")
        else:
            self.cvm_radius = None

    def configure_flow(self, config: Union[dict, None]) -> None:
        default_config = dict(
            dims=self.dims,
            n_neurons=max(2 * self.dims, 8),
            batch_norm_between_transforms=True,
        )
        if config:
            default_config.update(config)
        self.flow_config = default_config

    def train(
        self,
        samples: torch.Tensor,
        logl: torch.Tensor,
        reset: bool = True,
        validation_fraction: float = 0.2,
        patience: int = 20,
    ) -> None:
        if reset or self.flow is None:
            logger.debug("Resetting flow")
            self.flow = get_realnvp(**self.flow_config)
            self.flow = self.flow.to(self.device)

        logger.debug("Training flowproposal")

        max_epochs = 500
        optimizer = torch.optim.Adam(
            self.flow.parameters(),
            lr=5e-3,
            weight_decay=1e-6,
        )

        if self.logit:
            f = torch.logit
        else:
            f = lambda x: x

        self.scale = f(samples).std(dim=0)
        self.shift = f(samples).mean(dim=0)
        samples_prime, _ = self.rescale(samples)

        indices = torch.randperm(len(samples))
        samples_prime = samples_prime[indices, ...]

        n_train = int((1 - validation_fraction) * len(samples))
        train_samples, val_samples = (
            samples_prime[:n_train, ...],
            samples_prime[n_train:, ...],
        )

        history = dict(
            loss=TensorList(buffer_size=max_epochs, device=self.device),
            val_loss=TensorList(buffer_size=max_epochs, device=self.device),
        )
        best_val_loss = torch.inf
        best_it = 0
        best_state = None

        for it in range(max_epochs):
            optimizer.zero_grad()
            loss = -self.flow.log_prob(train_samples).mean()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.flow.parameters(), 10)

            optimizer.step()
            history["loss"].append(loss.detach())

            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss value")

            with torch.inference_mode():
                val_loss = -self.flow.log_prob(val_samples).mean()
            history["val_loss"].append(val_loss.detach())

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_it = it
                best_state = copy.deepcopy(self.flow.state_dict())

            if (it - best_it) > patience:
                logger.debug("Stopping training early")
                break

        if best_state:
            self.flow.load_state_dict(best_state)
        else:
            raise RuntimeWarning("Flow failed to train!")
        self.flow.eval()

    def rescale(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        log_j = -torch.log(self.scale).sum() * torch.ones(
            x.shape[0], device=x.device
        )
        if self.logit:
            x, lj = logit_with_jacobian(x)
            log_j += lj.sum(-1)
        x = (x - self.shift) / self.scale
        return x, log_j

    def rescale_inverse(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        log_j = torch.log(self.scale).sum() * torch.ones(
            x.shape[0], device=x.device
        )
        x = self.scale * x + self.shift
        if self.logit:
            x, lj = sigmoid_with_jacobian(x)
            log_j += lj.sum(-1)
        return x, log_j

    def _draw_latent_samples(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.cvm_radius:
            if self.sample_nball:
                z = sample_nball(
                    dims=self.dims,
                    n=batch_size,
                    radius=self.cvm_radius,
                    device=self.device,
                )
                log_q = torch.zeros(batch_size, device=self.device)
            else:
                z = sample_radially_truncated_gaussian(
                    dims=self.dims,
                    n=batch_size,
                    radius=self.cvm_radius,
                    device=self.device,
                    u_max=self.u_max,
                )
                # log_q is not normalized but that does not matter
                # because we normalize the weights in the rejection sampling
                log_q = self.flow._distribution.log_prob(z)
        else:
            z, log_q = self.flow._distribution.sample_and_log_prob(batch_size)
        return z, log_q

    def populate(
        self,
        live_points: torch.Tensor,
        logl: torch.Tensor,
        n: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> None:
        if n is None:
            n = self.poolsize

        if batch_size is None:
            batch_size = self.batch_size

        samples = torch.empty((0, self.dims), device=self.device)
        log_weights = torch.empty(0, device=self.device)
        log_n = math.log(n)
        log_m = -math.inf
        log_constant = -torch.inf
        n_accepted = 0
        n_proposed = 0

        if self.truncate_log_q:
            x, log_j = self.rescale(live_points)
            log_q_min = (self.flow.forward_and_log_prob(x)[1] + log_j).min()
        else:
            log_q_min = None

        self.flow.eval()

        while n_accepted < n:
            n_proposed += batch_size
            with torch.inference_mode():
                z, log_q = self._draw_latent_samples(batch_size)
                x, log_j = self.flow.inverse(z)
            log_q = log_q - log_j

            # Rescale
            x, log_j = self.rescale_inverse(x)
            log_q = log_q - log_j
            # Reject out-of-bounds
            ib = ~torch.any((x < 0.0) | (x > 1.0), dim=1)
            if not torch.any(ib):
                continue
            x, log_q = x[ib, ...], log_q[ib]

            if log_q_min:
                keep = log_q > log_q_min
                x, log_q = x[keep, ...], log_q[keep]

            # Rejection sampling
            # w = p / q
            log_w = -log_q

            if self.accumulate_weights:
                log_weights = torch.cat([log_weights, log_w])
                log_constant = max(log_constant, torch.max(log_w))
                samples = torch.cat([samples, x], dim=0)
                log_m = torch.logsumexp(log_weights - log_constant, -1)
                if log_m >= log_n:
                    log_u = torch.log(torch.rand_like(log_weights))
                    accept = (log_weights - log_constant) > log_u
                    n_accepted = accept.sum()
                if len(samples) >= self.max_samples:
                    logger.warning(
                        "Reached max samples (%s)", self.max_samples
                    )
                    log_u = torch.log(torch.rand_like(log_weights))
                    accept = (log_weights - log_constant) > log_u
                    n_accepted = accept.sum()
                    break
            else:
                log_w = log_w - log_w.max()
                log_u = torch.log(torch.rand_like(log_w))
                accept = log_w > log_u
                n_accept_batch = accept.sum()
                m = min(n - n_accepted, n_accept_batch)
                samples = torch.cat([samples, x[accept][:m]])
                n_accepted += m

        acceptance = n_accepted / n_proposed

        if self.accumulate_weights:
            self.samples = samples[accept][:n]
        else:
            self.samples = samples[:n]
        self.logl = None
        self.indices = list(range(len(self.samples)))
        self.count += 1
        self.populated = True
        return acceptance
