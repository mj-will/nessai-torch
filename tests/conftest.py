import pytest
import torch


@pytest.fixture(
    params=["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
)
def device(request):
    return torch.device(request.param)
