import matplotlib.pyplot as plt
from nessai_torch.plot import (
    get_default_figsize,
    plot_samples_1d,
)
import pytest
import torch


@pytest.fixture(autouse=True)
def close_plots():
    plt.close("all")


def test_get_default_figsize():
    figsize = get_default_figsize()
    assert len(figsize) == 2


def test_plot_samples_1d():
    samples = [
        torch.randn(10, 2),
        torch.randn(10, 2),
    ]
    plot_samples_1d(
        *samples,
        labels=["train", "val"],
        parameter_labels=["x_0", "x_1"],
    )
