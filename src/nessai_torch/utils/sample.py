from typing import Optional

from scipy import stats
import torch
from .gammaincinv import gammaincinv


def rejection_sample(
    samples: torch.Tensor, log_w: torch.Tensor
) -> torch.Tensor:
    """Perform rejection sampling given samples and log-weights."""
    log_u = torch.log(torch.rand_like(log_w))
    log_w = log_w - log_w.max()
    return samples[log_w > log_u]


def sample_nball(
    dims: int,
    n: int = 1,
    radius: float = 1.0,
    device: Optional[torch.DeviceObjType] = None,
) -> torch.Tensor:
    """Sample from an n-ball with radius r"""
    z = torch.randn(n, dims, device=device)
    z = z / torch.sqrt(torch.sum(z**2, dim=1, keepdim=True))
    r = radius * (torch.rand(n, 1, device=device) ** (1 / dims))
    return r * z


def sample_radially_truncated_gaussian(
    dims: int,
    n: int = 1,
    radius: float = 1.0,
    u_max: Optional[float] = None,
    device: Optional[torch.DeviceObjType] = None,
) -> torch.Tensor:
    """Sample from a radially truncated Gaussian.

    Truncation can be specified with a radius or the maximum value of the
    CDF (:code:`u_max`)

    Parameters
    ----------
    dims
        The number of dimensions.
    n
        The number of samples to draw.
    radius
        The radius at which to truncate the Gaussian. Ignored if :code:`u_max`
        is specified.
    device
        The device to use for the samples.

    Returns
    -------
    Tensor of samples with shape (n, dims)
    """
    if u_max is None:
        u_max = stats.chi(df=dims).cdf(radius.cpu().numpy())
    u = u_max * torch.rand(n, device=device)
    # Inverse CDF of a chi-distribution
    p = torch.sqrt(
        2 * gammaincinv(torch.tensor(0.5 * dims, device=u.device), u)
    )
    x = torch.randn(dims, n, device=device)
    points = (p * x / torch.sqrt(torch.sum(torch.square(x), dim=0))).T
    return points
