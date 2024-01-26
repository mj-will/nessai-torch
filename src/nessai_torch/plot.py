"""Plotting functions"""
import logging
from typing import List, Optional

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import torch

logger = logging.getLogger(__name__)


def save_figure(
    figure: Figure, filename: Optional[str] = None
) -> Optional[Figure]:
    if filename is not None:
        figure.tight_layout()
        figure.savefig(filename)
        plt.close(figure)
    else:
        return figure


def get_default_figsize() -> list:
    return list(plt.rcParams["figure.figsize"])


def plot_trace(
    logx: torch.Tensor,
    nested_samples: torch.Tensor,
    labels: Optional[List[str]] = None,
    filename: Optional[str] = None,
) -> Optional[Figure]:
    """Produce a trace plot."""

    logx = logx.cpu().numpy()
    nested_samples = nested_samples.cpu().numpy()

    dims = nested_samples.shape[-1]

    figsize = (5, dims * 2)

    fig, axs = plt.subplots(dims, 1, sharex=True, figsize=figsize)

    if labels is None:
        labels = [f"x_{i}" for i in range(dims)]

    for i, samples in enumerate(nested_samples.T):
        axs[i].plot(logx, samples, ls="", marker=",")
        axs[i].set_ylabel(labels[i])

    axs[-1].set_xlabel(r"$\log X$")
    axs[-1].invert_xaxis()

    return save_figure(fig, filename)


def plot_insertion_indices(
    indices: torch.Tensor,
    nlive: int,
    confidence_intervals: Optional[List[float]] = None,
    filename: Optional[str] = None,
) -> Optional[Figure]:
    """Plot the insertion indices"""
    indices = indices.cpu().numpy()

    x = np.arange(0, nlive, 1)
    analytic = x / x[-1]

    n = len(indices)
    counts = np.bincount(indices, minlength=nlive)
    estimated = np.cumsum(counts) / n

    if confidence_intervals is None:
        confidence_intervals = [
            0.997,
        ]

    fig, axs = plt.subplots(2, 1, sharex=True)

    axs[0].hist(indices, bins="auto", histtype="step")

    axs[1].plot(x, analytic - estimated)

    for ci in confidence_intervals:
        bound = (1 - ci) / 2
        bound_values = stats.binom.ppf(1 - bound, n, analytic) / n
        lower = bound_values - analytic
        upper = analytic - bound_values
        upper[0] = 0
        upper[-1] = 0
        lower[0] = 0
        lower[-1] = 0
        axs[1].fill_between(x, lower, upper, color="grey", alpha=0.2)

    axs[1].set_xlabel("Insertion index")
    axs[1].set_ylabel("Analytic - estimated")

    axs[1].set_xlim(0, nlive - 1)

    return save_figure(fig, filename)


def corner_plot(
    samples: torch.Tensor,
    labels: Optional[list[str]] = None,
    filename: Optional[str] = None,
    **kwargs,
) -> Optional[Figure]:
    try:
        import corner
    except ImportError as e:
        logger.error(
            f"Could not import corner, skipping corner plot! Error: {e}"
        )
        return
    samples = samples.cpu().numpy()
    fig = corner.corner(samples, labels=labels, **kwargs)
    return save_figure(fig, filename)


def plot_samples_1d(
    *samples: torch.Tensor,
    labels: Optional[list[str]] = None,
    parameter_labels: Optional[list[str]],
    filename: Optional[str] = None,
    **kwargs,
) -> Optional[Figure]:
    """Plot 1d histograms for one or more sets of samples."""
    hist_kwargs = dict(
        density=True,
        histtype="step",
    )
    if kwargs:
        hist_kwargs.update(kwargs)

    if not labels:
        labels = [f"samples_{i}" for i in range(len(samples))]

    dims = samples[0].shape[-1]
    figsize = get_default_figsize()
    figsize[0] /= 2
    figsize[1] *= dims / 2

    fig, axs = plt.subplots(dims, 1, figsize=figsize)

    for array, label in zip(samples, labels):
        for ax, array_1d in zip(axs, array.T):
            ax.hist(array_1d.detach(), label=label, **hist_kwargs)
    axs[-1].legend()

    if parameter_labels:
        for ax, label in zip(axs, parameter_labels):
            ax.set_xlabel(label)

    return save_figure(fig, filename)
