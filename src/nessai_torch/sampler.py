import logging
import os
from typing import Callable, Optional

import matplotlib.pyplot as plt
import torch

from .evidence import EvidenceIntegral
from .proposal.base import ProposalWithPool
from .proposal.flow import FlowProposal
from .utils.sample import rejection_sample


logger = logging.getLogger(__name__)


class Sampler:
    def __init__(
        self,
        log_likelihood: Callable,
        prior_transform: Callable,
        dims: int,
        nlive: int = 1000,
        tolerance: float = 0.1,
        outdir: Optional[str] = None,
        plot_pool: bool = False,
        proposal_class: Optional[Callable] = None,
        reset_flow: int = 1,
        device: str = "cpu",
        **kwargs,
    ) -> None:
        self.log_likelihood = log_likelihood
        self.prior_transform = prior_transform
        self.nlive = nlive
        self.dims = dims
        self.tolerance = tolerance
        self.iteration = 0
        self.outdir = os.getcwd() if outdir is None else outdir
        self.plot_pool = plot_pool

        self.reset_flow = int(reset_flow)
        self.populate_count = 0
        self.device = torch.device(device)

        os.makedirs(self.outdir, exist_ok=True)

        self._nested_samples = []
        self.acceptance = []
        self.indices = []
        self.live_points = None
        self.logl = None
        self.logl_min = -torch.inf
        self.logl_max = -torch.inf
        self.criterion = torch.inf
        self.finalised = False

        self.integral = EvidenceIntegral(nlive=self.nlive, device=self.device)

        if proposal_class is None:
            proposal_class = FlowProposal

        if (
            issubclass(proposal_class, ProposalWithPool)
            and "poolsize" not in kwargs
        ):
            kwargs["poolsize"] = self.nlive

        self.proposal = proposal_class(
            dims=self.dims,
            device=self.device,
            **kwargs,
        )

    @property
    def stop(self) -> bool:
        if self.criterion < self.tolerance:
            return True
        else:
            return False

    @property
    def nested_samples(self) -> torch.tensor:
        ns = torch.stack(self._nested_samples, dim=0).detach()
        ns.requires_grad_(False)
        return ns

    @property
    def posterior_samples(self) -> torch.Tensor:
        return rejection_sample(
            self.nested_samples, self.integral.log_posterior_weights
        )

    def initialise(self) -> None:
        self.live_points = torch.rand(
            (self.nlive, self.dims), device=self.device
        ).requires_grad_(False)
        logl = self.log_likelihood(self.prior_transform(self.live_points))
        idx = torch.argsort(logl)
        self.live_points = self.live_points[idx]
        self.logl = logl[idx]

    def finalise(self) -> None:
        """Finalise the nested sampling run.

        This includes consuming the final live points
        """
        for (i, x), logl in zip(enumerate(self.live_points), self.logl):
            self.integral.update(logl, self.nlive - i)
            self._nested_samples.append(x.detach())
        self.live_points = None

        self.plot()

        logger.info(f"Final log-evidence: {self.integral.logz:.3f}")

        self.finalised = True

    def insert_live_point(self, x, logl):
        """
        Insert a live point
        """
        # This is the index including the current worst point, so final index
        # is one less, otherwise index=0 would never be possible
        index = torch.searchsorted(self.logl, logl)
        self.live_points[: index - 1, :] = self.live_points[1:index, :].clone()
        self.live_points[index - 1, :] = x
        self.logl.data[: index - 1] = self.logl[1:index].clone()
        self.logl[index - 1] = logl
        return index

    def step(self) -> None:
        """Perform one nested sampling step"""
        self.logl_min = self.logl[0].clone()
        self.integral.update(self.logl_min, self.nlive)
        self._nested_samples.append(self.live_points[0].detach().clone())
        count = 0
        while True:
            if not self.proposal.populated:
                self.proposal.populate(
                    self.live_points,
                    self.logl,
                    reset=self.reset_flow
                    and (not bool(self.populate_count % self.reset_flow)),
                )
                self.proposal.compute_likelihoods(
                    self.log_likelihood,
                    self.prior_transform,
                )
                if self.plot_pool:
                    self.proposal.plot(self.outdir)
                self.populate_count += 1
            x, logl = self.proposal.draw(self.live_points[0])
            if logl is None:
                logl = self.log_likelihood(self.prior_transform(x))
            count += 1
            if logl > self.logl_min:
                break
        index = self.insert_live_point(x, logl)
        self.indices.append(index.item())
        self.acceptance.append(1 / count)
        self.logl_max = self.logl[-1]
        self.logl_min = self.logl[0]

    def update_criterion(self) -> None:
        self.criterion = (
            torch.logaddexp(
                self.integral.logz, self.logl_max - self.iteration / self.nlive
            )
            - self.integral.logz
        )

    def plot(self) -> None:
        fig, axs = plt.subplots(self.dims, 1, sharex=True)

        ns = self.nested_samples.cpu()
        for i in range(self.dims):
            axs[i].scatter(self.integral.logx.data[1:].cpu(), ns[:, i], s=1.0)

        axs[-1].set_xlabel(r"$\log X$")

        fig.savefig(os.path.join(self.outdir, "trace.png"))

        fig = plt.figure()
        plt.plot(self.acceptance, ",")
        plt.xlabel("Iteration")
        plt.ylabel("Acceptance")
        fig.savefig(os.path.join(self.outdir, "acceptance.png"))

        fig = plt.figure()
        plt.hist(self.indices, 20, density=True, histtype="step")
        fig.savefig(os.path.join(self.outdir, "indices.png"))
        plt.close("all")

    def run(self) -> None:
        self.initialise()

        while not self.stop:
            self.step()
            self.iteration += 1
            self.update_criterion()
            if self.iteration % (self.nlive // 10) == 0:
                logger.info(
                    f"it {self.iteration} - log Z={self.integral.logz:.2f}, log dZ={self.criterion:.2f}"
                )
            if self.iteration % (self.nlive) == 0:
                self.plot()

        self.finalise()
