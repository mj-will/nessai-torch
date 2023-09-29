import copy
import logging
import math
import os
from typing import Optional, Tuple, Union

import glasflow
import matplotlib.pyplot as plt
from scipy import stats
import torch

from .base import ProposalWithPool

from ..flows import get_realnvp
from ..tensorlist import TensorList
from ..transforms import logit_with_jacobian, sigmoid_with_jacobian
from ..utils.sample import sample_nball

logger = logging.getLogger(__name__)


class FlowProposal(ProposalWithPool):
    """Proposal that draws samples using a normalizing flow"""

    flow: glasflow.flows.base.Flow = None
    scale: torch.Tensor = None
    shift: torch.Tensor = None
    flow_config: dict = None

    def __init__(
        self,
        *,
        dims: int,
        device: torch.DeviceObjType,
        poolsize: int,
        batch_size: int = 1000,
        logit: bool = False,
        constant_volume_fraction: Optional[float] = None,
        truncate_log_q: bool = False,
        flow_config: Optional[dict] = None,
    ) -> None:
        super().__init__(dims=dims, device=device)

        self.poolsize = poolsize
        self.batch_size = batch_size
        self.truncate_log_q = truncate_log_q
        self.logit = logit
        self.count = 0

        self.configure_flow(flow_config)

        self.constant_volume_fraction = constant_volume_fraction
        if self.constant_volume_fraction is not None:
            self.cvm_radius = torch.tensor(
                stats.chi.ppf(self.constant_volume_fraction, self.dims)
            )
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

        self.flow.load_state_dict(best_state)
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

    def plot(self, outdir: str) -> None:
        fig = plt.figure()
        plt.scatter(self.samples[:, 0].detach(), self.samples[:, 1].detach())
        fig.savefig(os.path.join(outdir, f"pool_{self.count}.png"))
        plt.close()

    def sample_truncated_gaussian(self, n: int) -> torch.Tensor:
        """Draw n samples from a radially truncated Gaussian.

        Parameters
        ----------
        n : int
            Number of samples
        radius : float
            Radius

        Returns
        -------
        torch.Tensor
            Samples from the radially truncated Gaussian
        """

    def populate(
        self,
        live_points: torch.Tensor,
        logl: torch.Tensor,
        n: Optional[int] = None,
        batch_size: Optional[int] = None,
        **kwargs,
    ) -> None:
        self.train(live_points, logl, **kwargs)

        if n is None:
            n = self.poolsize

        if batch_size is None:
            batch_size = self.batch_size

        samples = torch.empty((n, self.dims), device=self.device)
        m = 0

        if self.truncate_log_q:
            x, log_j = self.rescale(live_points)
            log_q_min = (self.flow.forward_and_log_prob(x)[1] + log_j).min()
        else:
            log_q_min = None

        self.flow.eval()

        with torch.inference_mode():
            while m < n:
                # Sample
                if self.cvm_radius:
                    z = sample_nball(
                        dims=self.dims,
                        n=n,
                        radius=self.cvm_radius,
                        device=self.device,
                    )
                    log_q = torch.zeros(n, device=self.device)
                else:
                    z, log_q = self.flow._distribution.sample_and_log_prob(n)

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
                # w = p / (q / max(q))
                log_w = -log_q + log_q.min()
                log_u = torch.log(torch.rand_like(log_w))
                accept = log_w > log_u
                if not torch.any(accept):
                    continue
                # Add to samples
                k = min(accept.sum(), n - m)
                samples[m : m + k] = x[accept][:k]
                m += k

        self.samples = samples
        self.logl = None
        self.indices = list(range(n))
        self.count += 1
        self.populated = True
