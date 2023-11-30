from nessai_torch.utils.sample import sample_radially_truncated_gaussian
import pytest
from scipy import stats
import torch


@pytest.fixture(params=[2, 4, 8])
def dims(request):
    return request.param


@pytest.fixture(params=[1, 10, 100])
def n(request):
    return request.param


@pytest.fixture(params=torch.tensor([1.0, 4.0, 16.0, 64.0]))
def radius(request):
    return request.param


def test_truncated_gaussian_radius(dims, n, radius, device):
    x = sample_radially_truncated_gaussian(
        dims=dims, n=n, radius=radius, device=device
    )
    r = torch.sqrt(torch.sum(x**2, dim=-1))
    assert torch.all(r <= radius)


def test_truncated_gaussian_u_max(dims, n, radius, device):
    u_max = stats.chi(df=dims).cdf(radius.item())
    x = sample_radially_truncated_gaussian(
        dims=dims, n=n, u_max=u_max, device=device
    )
    r = torch.sqrt(torch.sum(x**2, dim=-1))
    assert torch.all(r <= radius)
