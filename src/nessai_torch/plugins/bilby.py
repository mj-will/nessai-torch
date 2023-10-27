"""Bilby wrapper for nessai_torch"""
import copy
from bilby.core.sampler.base_sampler import NestedSampler
import torch

from ..sampler import Sampler
from ..utils.logging import configure_logger


class NessaiTorch(NestedSampler):
    """Wrapper for nessai_torch.

    Implementation: https://github.com/mj-will/nessai-torch
    """

    external_sampler_name = "nessai_torch"
    kwargs: dict = None
    torch_dtype: torch.dtype = None

    def log_likelihood_wrapper(self, theta):
        """Wrapper for the log-likelihood.

        Note: nessai_torch requires a vectorized likelihood.
        """
        theta = theta.cpu().numpy()
        logl = []
        for t in theta:
            logl.append(self.log_likelihood(t))
        return torch.tensor(logl, device=self._sampler.device)

    def prior_transform_wrapper(self, theta):
        """Wrapper for the prior transform.

        Note: nessai_torch requires this be vectorized.
        """
        out = []
        for t in theta:
            out.append(self.prior_transform(t))
        return torch.tensor(out, device=self._sampler.device)

    def run_sampler(self):
        """Run the sampler"""

        kwargs = copy.deepcopy(self.kwargs)
        if kwargs is None:
            kwargs = {}

        self.torch_dtype = torch.get_default_dtype()

        configure_logger()

        self._sampler = Sampler(
            log_likelihood=self.log_likelihood_wrapper,
            prior_transform=self.prior_transform_wrapper,
            dims=self.ndim,
            outdir=self.outdir,
            **kwargs,
        )

        self._sampler.run()

        self.result.samples = self._sampler.posterior_samples.cpu().numpy()
        self.result.nested_samples = self._sampler.nested_samples.cpu().numpy()
        self.result.log_evidence = self._sampler.log_evidence.item()
        return self.result
