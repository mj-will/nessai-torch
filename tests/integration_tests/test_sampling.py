import pytest
import torch

from nessai_torch.sampler import Sampler
from nessai_torch.utils.bounds import in_bounds
from nessai_torch.proposal.flow import FlowProposal
from nessai_torch.proposal.prior import PriorProposal
from nessai_torch.proposal.spherical import SphericalProposal


@pytest.fixture
def dims():
    return 2


@pytest.fixture
def sampler_inputs(dims, device):
    mean = torch.zeros(dims, device=device)
    cov = torch.eye(dims, device=device)
    bounds = torch.tensor([-10.0, 10.0], device=device)
    dist = torch.distributions.MultivariateNormal(mean, cov)

    def log_likelihood(x: torch.tensor) -> torch.tensor:
        return dist.log_prob(x) + torch.log(in_bounds(x, bounds))

    def prior_transform(x: torch.tensor) -> torch.tensor:
        return (bounds[1] - bounds[0]) * x + bounds[0]

    return dims, log_likelihood, prior_transform


@pytest.mark.parametrize(
    "ProposalClass, kwargs",
    [
        (
            FlowProposal,
            {"sample_nball": False, "constant_volume_fraction": 0.95},
        ),
        (
            FlowProposal,
            {"sample_nball": True, "constant_volume_fraction": 0.95},
        ),
        (FlowProposal, {}),
        (SphericalProposal, {}),
        (PriorProposal, {}),
    ],
)
def test_sampling(sampler_inputs, device, ProposalClass, tmp_path, kwargs):
    dims, log_likelihood, prior_transform = sampler_inputs

    outdir = tmp_path / "test_proposal"
    outdir.mkdir()

    sampler = Sampler(
        log_likelihood=log_likelihood,
        prior_transform=prior_transform,
        dims=dims,
        device=device,
        outdir=outdir,
        nlive=50,
        tolerance=5.0,
        proposal_class=ProposalClass,
        **kwargs,
    )
    sampler.run()
