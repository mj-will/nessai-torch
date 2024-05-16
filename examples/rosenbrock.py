"""Basic example using the Rosenbrock likelihood."""

import os

import torch

from nessai_torch.sampler import Sampler
from nessai_torch.plot import corner_plot
from nessai_torch.utils.bounds import in_bounds
from nessai_torch.utils.logging import configure_logger

# Limit torch to a single thread
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

logger = configure_logger("INFO")

dims = 4
bounds = torch.tensor([-5.0, 5.0])
outdir = "outdir_rosenbrock"


@torch.jit.script
def rosenbrock(x: torch.Tensor) -> torch.Tensor:
    r"""Rosenbrock function in N dimensions.

    This is the more involved variant given by

    .. math::
        \sum_{i=1}^{N-1} [100(x_{i+1} - x_{i}^{2})^2 + (1 - x_{i})^2].
    """
    return torch.sum(
        100.0 * (x[..., 1:] - x[..., :-1] ** 2.0) ** 2.0
        + (1.0 - x[..., :-1]) ** 2.0,
        dim=-1,
    )


# Define the log-likelihood
def log_likelihood(x: torch.Tensor) -> torch.Tensor:
    return torch.log(in_bounds(x, bounds)) - rosenbrock(x)


# Define the mapping from the unit hyper-cube to the prior
def prior_transform(x: torch.Tensor) -> torch.Tensor:
    return (bounds[1] - bounds[0]) * x + bounds[0]


# Initialise the sampler
sampler = Sampler(
    log_likelihood=log_likelihood,
    prior_transform=prior_transform,
    dims=dims,
    outdir=outdir,
    nlive=1_000,
    constant_volume_fraction=0.98,
    flow_config=dict(n_neurons=32, batch_norm_between_transforms=True),
    reset_flow=4,
)

# Run the sampler
sampler.run()

# Plot the posterior samples
corner_plot(
    sampler.posterior_samples, filename=os.path.join(outdir, "posterior.png")
)
