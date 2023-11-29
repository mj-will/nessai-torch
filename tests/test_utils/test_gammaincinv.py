from nessai_torch.utils.gammaincinv import gammaincinv
import pytest
from scipy.special import gammaincinv as gammaincinv_scipy
import torch

# Call once to compile
gammaincinv(torch.tensor(1.0), torch.tensor(1.0))


# The tests fail outside these limits
@pytest.mark.parametrize("a", torch.linspace(1.5, 35, 32))
def test_gammaincinv_invertible(a):
    x = torch.linspace(0.0, 10.0, 10)
    y = torch.special.gammainc(a, x)
    x_out = gammaincinv(a, y)
    assert torch.isclose(x_out, x).all()


@pytest.mark.parametrize("a", torch.linspace(0.1, 101, 50))
def test_gammaincinv_scipy(a):
    y = torch.linspace(0, 1, 1000)
    x_out = gammaincinv(a, y)
    x_expected = gammaincinv_scipy(a.numpy(), y.numpy())
    assert torch.isclose(x_out, torch.tensor(x_expected)).all()
