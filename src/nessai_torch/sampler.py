import logging
import os
import time
from typing import Callable, Optional

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import torch

from .evidence import EvidenceIntegral
from .plot import (
    plot_trace,
    plot_insertion_indices,
    save_figure,
    plot_samples_1d,
)
from .proposal.base import ProposalWithPool
from .proposal.flow import FlowProposal
from .proposal.prior import PriorProposal
from .utils.sample import rejection_sample
from .utils.stats import rolling_mean_numpy
from .utils.io import save_dict_to_hdf5
from .tensorlist import TensorList


logger = logging.getLogger(__name__)


class Sampler:
    def __init__(
        self,
        log_likelihood: Callable,
        prior_transform: Callable,
        dims: int,
        nlive: int = 1000,
        tolerance: float = 0.1,
        sample_prior_iterations: int = 2000,
        outdir: Optional[str] = None,
        save: bool = True,
        parameter_labels: Optional[list[str]] = None,
        plot_pool: bool = False,
        plot_trace: bool = True,
        plot_insertion_indices: bool = True,
        plot_state: bool = True,
        proposal_class: Optional[Callable] = None,
        reset_flow: int = 0,
        device: str = "cpu",
        seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        self.log_likelihood = log_likelihood
        self.prior_transform = prior_transform
        self.nlive = nlive
        self.dims = dims
        self.tolerance = tolerance
        self.iteration = 0
        self.outdir = os.getcwd() if outdir is None else outdir
        self.save = save
        self.plot_pool = plot_pool
        self._plot_trace = plot_trace
        self._plot_insertion_indices = plot_insertion_indices
        self._plot_state = plot_state
        self.parameter_labels = parameter_labels
        self.seed = seed

        self.reset_flow = int(reset_flow)
        self.sample_prior_iterations = sample_prior_iterations
        self.populate_count = 0
        self.n_likelihood_calls = 0
        self.sampling_time = 0
        self.device = torch.device(device)

        logger.info(f"Running with device={self.device}")

        os.makedirs(self.outdir, exist_ok=True)

        if self.seed:
            logger.info(f"Setting random seed to: {self.seed}")
            torch.manual_seed(self.seed)

        self._nested_samples = TensorList(
            size=(self.dims,), device=self.device
        )
        self.history = dict(
            acceptance=[],
            proposal_acceptance=[],
        )
        self.indices = TensorList(device=self.device, dtype=torch.int)
        self._logl_nested_samples = TensorList(device=self.device)
        self.live_points = None
        self.logl = None
        self.logl_min = -torch.inf
        self.logl_max = -torch.inf
        self.criterion = torch.inf
        self.finalised = False

        self.integral = EvidenceIntegral(nlive=self.nlive, device=self.device)

        if proposal_class is None:
            proposal_class = FlowProposal

        if proposal_class.has_pool and "poolsize" not in kwargs:
            kwargs["poolsize"] = self.nlive

        self.proposal = proposal_class(
            dims=self.dims,
            device=self.device,
            log_likelihood_fn=self.log_likelihood_unit_hypercube,
            **kwargs,
        )

        self.prior_proposal = PriorProposal(
            dims=self.dims,
            device=self.device,
            log_likelihood_fn=self.log_likelihood_unit_hypercube,
        )

    @property
    def stop(self) -> bool:
        if self.criterion < self.tolerance:
            return True
        else:
            return False

    @property
    def nested_samples(self) -> torch.Tensor:
        return self.prior_transform(self._nested_samples.data)

    @property
    def log_likelihoods_nested_samples(self) -> torch.Tensor:
        return self._logl_nested_samples.data

    @property
    def posterior_samples(self) -> torch.Tensor:
        return rejection_sample(
            self.nested_samples,
            self.log_posterior_weights,
        )

    @property
    def log_posterior_weights(self) -> torch.Tensor:
        return self.integral.log_posterior_weights

    @property
    def log_evidence(self) -> torch.Tensor:
        return self.integral.logz

    @property
    def log_evidence_error(self) -> torch.Tensor:
        return self.integral.logz_error

    @property
    def insertion_indices(self) -> torch.Tensor:
        return self.indices.data

    @property
    def should_reset(self) -> bool:
        """Boolean that indicates if the proposal should be reset"""
        return self.reset_flow and not bool(
            self.populate_count % self.reset_flow
        )

    def log_likelihood_unit_hypercube(self, x: torch.Tensor) -> torch.Tensor:
        """Log-likelihood for samples in the unit hypercube"""
        self.n_likelihood_calls += len(x)
        return self.log_likelihood(self.prior_transform(x))

    def initialise(self) -> None:
        live_points, logl = list(
            zip(*[self.prior_proposal.draw(None) for _ in range(self.nlive)])
        )
        live_points = torch.cat(live_points, dim=0)
        logl = torch.cat(logl)
        idx = torch.argsort(logl)
        self.live_points = live_points[idx]
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

        logger.info(
            f"Final log-evidence: {self.integral.logz:.3f} +/- "
            f"{self.integral.logz_error:.3f}"
        )

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
        return index - 1

    def update_proposal(self) -> None:
        """Update the proposal.

        This includes training and populating the proposal.
        """
        if self.proposal.has_pool:
            if not self.proposal.populated:
                if self.proposal.trainable:
                    with torch.enable_grad():
                        self.proposal.train(
                            self.live_points,
                            self.logl,
                            reset=self.should_reset,
                        )
                proposal_acceptance = self.proposal.populate(
                    self.live_points,
                    self.logl,
                )
                self.history["proposal_acceptance"].append(
                    (self.iteration, proposal_acceptance)
                )
                self.proposal.compute_likelihoods()
                if self.plot_pool:
                    plot_samples_1d(
                        self.live_points,
                        self.proposal.samples,
                        labels=["live points", "pool"],
                        parameter_labels=self.parameter_labels,
                        filename=os.path.join(
                            self.outdir, f"pool_it_{self.iteration}.png"
                        ),
                    )
                self.populate_count += 1
        elif self.proposal.trainable:
            with torch.enable_grad():
                self.proposal.train(
                    self.live_points,
                    self.logl,
                )

    def step(self) -> None:
        """Perform one nested sampling step"""
        self.logl_min = self.logl[0].clone()
        self.integral.update(self.logl_min, self.nlive)
        self._nested_samples.append(self.live_points[0].detach().clone())
        self._logl_nested_samples.append(self.logl[0].detach().clone())
        count = 0
        while True:
            if self.iteration < self.sample_prior_iterations:
                x, logl = self.prior_proposal.draw(self.live_points[0])
            else:
                self.update_proposal()
                x, logl = self.proposal.draw(self.live_points[0])
            if logl is None:
                logl = self.log_likelihood_unit_hypercube(x)
            count += 1
            if torch.isfinite(logl) and logl > self.logl_min:
                break
        index = self.insert_live_point(x, logl)
        self.indices.append(index)
        self.history["acceptance"].append(1 / count)
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
        if self._plot_trace:
            plot_trace(
                self.integral.logx.data[1:],
                self.nested_samples,
                labels=self.parameter_labels,
                filename=os.path.join(self.outdir, "trace.png"),
            )

        if self._plot_insertion_indices:
            plot_insertion_indices(
                self.indices.data,
                nlive=self.nlive,
                filename=os.path.join(self.outdir, "insertion_indices.png"),
            )

        if self._plot_state:
            self.plot_state(os.path.join(self.outdir, "state.png"))

    def plot_state(self, filename: Optional[str] = None) -> Optional[Figure]:
        n_plots = 3

        fig, axs = plt.subplots(n_plots, 1, sharex=True)

        its = np.arange(self.iteration)

        j = 0

        # logl will be longer than its after finalizing the run
        axs[j].plot(self.integral.logl.data[1:].cpu().numpy())
        axs[j].set_ylabel(r"$\log L^{*}$")
        j += 1

        axs[j].plot(its, self.history["acceptance"], marker=",", ls="")
        axs[j].plot(
            its,
            rolling_mean_numpy(self.history["acceptance"], self.nlive // 10),
            color="C0",
        )
        axs[j].set_ylabel("Acceptance")
        axs[j].set_yscale("log")
        j += 1
        if len(self.history["proposal_acceptance"]):
            proposal_its, proposal_acceptance = np.array(
                self.history["proposal_acceptance"]
            ).T
            axs[j].plot(proposal_its, proposal_acceptance, marker=".", ls="")
        axs[j].set_ylabel("Proposal acceptance")
        axs[j].set_yscale("log")

        axs[-1].set_xlabel("Iteration")
        return save_figure(fig, filename)

    @torch.no_grad()
    def run(self) -> None:
        start = time.perf_counter()
        self.initialise()

        while not self.stop:
            self.step()
            self.iteration += 1
            self.update_criterion()
            if self.iteration % (self.nlive // 10) == 0:
                logger.info(
                    f"it {self.iteration} - "
                    f"log Z={self.integral.logz:.2f} +/- "
                    f"{self.integral.logz_error:.2f}, "
                    f"log dZ={self.criterion:.2f}, "
                    f"H={self.integral.info:.2f}"
                )
            if self.iteration % (self.nlive) == 0:
                self.plot()

        self.finalise()
        stop = time.perf_counter()
        self.sampling_time += stop - start
        logger.info(f"Total likelihood evaluations: {self.n_likelihood_calls}")
        logger.info(f"Total sampling time : {self.sampling_time:.1f} s")

        if self.save:
            self.save_result()

    def get_result_dictionary(self) -> dict:
        """Return a dictionary containing the results"""
        results = dict()
        results["nested_samples"] = self.nested_samples
        results["log_likelihoods"] = self.log_likelihoods_nested_samples
        results["log_posterior_weights"] = self.log_posterior_weights
        results["log_evidence"] = self.log_evidence
        results["log_evidence_error"] = self.log_evidence_error
        results["criterion"] = self.criterion
        results["insertion_indices"] = self.insertion_indices
        results["n_likelihood_calls"] = self.n_likelihood_calls
        results["sampling_time"] = self.sampling_time
        return results

    def save_result(self, filename: Optional[str] = None) -> None:
        """Save the results to a file.

        Defaults to :code:`result.hdf5` if the filename is not specified.
        """
        if filename is None:
            filename = os.path.join(self.outdir, "result.hdf5")
        save_dict_to_hdf5(self.get_result_dictionary(), filename)
