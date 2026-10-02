"""MCMC convergence diagnostics."""

from __future__ import annotations

import numpy as np


def gelman_rubin(chains: np.ndarray) -> np.ndarray:
    """Gelman–Rubin diagnostic for multiple chains.

    Parameters
    ----------
    chains : np.ndarray, shape (m, n, d)
        m chains, n draws, d parameters.

    Returns
    -------
    rhat : np.ndarray, shape (d,)
    """
    m, n, d = chains.shape
    chain_means = chains.mean(axis=1)                 # (m, d)
    chain_vars = chains.var(axis=1, ddof=1)           # (m, d)
    B = n * chain_means.var(axis=0, ddof=1)           # (d,)
    W = chain_vars.mean(axis=0)                       # (d,)
    var_hat = (n - 1) / n * W + B / n
    rhat = np.sqrt(var_hat / W)
    return rhat


def effective_sample_size(chains: np.ndarray) -> np.ndarray:
    """Approximate effective sample size using the initial-positive-sequence
    autocorrelation estimator.

    Parameters
    ----------
    chains : np.ndarray, shape (m, n, d)

    Returns
    -------
    ess : np.ndarray, shape (d,)
    """
    m, n, d = chains.shape
    ess = np.zeros(d)
    for j in range(d):
        # Concatenate chains
        x = chains[:, :, j].reshape(-1)
        n_total = len(x)
        x = x - x.mean()
        # Autocovariance
        acov = np.correlate(x, x, mode="full")[n_total - 1:] / n_total
        # Sum autocorrelations until first negative
        rho = acov / acov[0]
        cutoff = 1
        for k in range(1, len(rho)):
            if rho[k] < 0:
                break
            cutoff = k
        tau = 1 + 2 * rho[1:cutoff + 1].sum()
        ess[j] = n_total / max(tau, 1e-6)
    return ess