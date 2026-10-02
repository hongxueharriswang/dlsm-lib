"""Visualization helpers for the DLSM."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt


def plot_latent_trajectories(
    true_states: Optional[np.ndarray],
    filtered_mean: np.ndarray,
    filtered_std: Optional[np.ndarray] = None,
    components: Sequence[int] = (0, 1, 2, 3, 4, 5),
    labels: Sequence[str] = ("e^O", "e^P", "e^C", "e^SR", "Progress", "Risk"),
    figsize: tuple = (12, 8),
):
    """Plot latent trajectories for a single student."""
    n = len(components)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True)
    axes = np.atleast_2d(axes)
    T = filtered_mean.shape[0]
    t = np.arange(T)

    for k, comp in enumerate(components):
        ax = axes[k // ncols, k % ncols]
        if true_states is not None:
            ax.plot(t, true_states[:, comp], "k-", label="True", lw=1.5)
        ax.plot(t, filtered_mean[:, comp], "C0-", label="Filtered mean")
        if filtered_std is not None:
            lo = filtered_mean[:, comp] - 1.96 * filtered_std[:, comp]
            hi = filtered_mean[:, comp] + 1.96 * filtered_std[:, comp]
            ax.fill_between(t, lo, hi, color="C0", alpha=0.2, label="95% CI")
        ax.set_ylabel(labels[comp])
        ax.legend(loc="best", fontsize=8)
    axes[-1, 0].set_xlabel("Time")
    axes[-1, 1].set_xlabel("Time")
    fig.tight_layout()
    return fig


def plot_status_probabilities(
    status_probs: np.ndarray,
    true_status: Optional[np.ndarray] = None,
    labels: Sequence[str] = ("Registered", "Progressing", "Stopped-Out", "Withdrawn", "Completed"),
    figsize: tuple = (10, 4),
):
    """Plot posterior status probabilities over time."""
    fig, ax = plt.subplots(figsize=figsize)
    T = status_probs.shape[0]
    t = np.arange(T)
    bottom = np.zeros(T)
    for s, lab in enumerate(labels):
        ax.bar(t, status_probs[:, s], bottom=bottom, label=lab, width=0.8)
        bottom += status_probs[:, s]
    if true_status is not None:
        ax.plot(t, true_status, "k.", markersize=3, label="True")
    ax.set_xlabel("Time")
    ax.set_ylabel("Posterior probability")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def plot_posterior(
    chain: np.ndarray,
    param_names: Sequence[str],
    true_params: Optional[np.ndarray] = None,
    figsize: tuple = (12, 10),
):
    """Plot marginal posterior distributions for each parameter."""
    d = chain.shape[1]
    ncols = 3
    nrows = int(np.ceil(d / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_2d(axes)
    for j in range(d):
        ax = axes[j // ncols, j % ncols]
        ax.hist(chain[:, j], bins=30, density=True, alpha=0.7, color="C0")
        if true_params is not None:
            ax.axvline(true_params[j], color="k", lw=2, label="True")
        ax.set_title(param_names[j])
        ax.legend(fontsize=8)
    for j in range(d, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")
    fig.tight_layout()
    return fig


def plot_calibration(
    predicted_prob: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 10,
    figsize: tuple = (5, 5),
):
    """Plot a calibration curve."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.digitize(predicted_prob, bins) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    observed = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() > 0:
            observed[b] = outcomes[mask].mean()
            counts[b] = mask.sum()
    centers = (bins[:-1] + bins[1:]) / 2
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1], "k--", label="Perfect")
    ax.plot(centers, observed, "o-", label="Observed")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.legend()
    fig.tight_layout()
    return fig