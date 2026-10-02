"""Observation model for the DLSM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import special
from scipy.stats import poisson, bernoulli

from dlsm.core.parameters import DLSParams


@dataclass
class Observation:
    """A single observation record for one student at one time step.

    Any of the fields may be ``None`` to indicate a missing channel.
    """

    counts: Optional[np.ndarray] = None       # shape (K_count,)
    binaries: Optional[np.ndarray] = None     # shape (K_binary,)
    ordinals: Optional[np.ndarray] = None     # shape (K_ordinal,)
    events: Optional[np.ndarray] = None       # shape (4,) binary indicators


class ObservationModel:
    """Heterogeneous observation model.

    Channels:
    - Poisson counts:  y_k ~ Poisson( exp(w_k^T x) )
    - Bernoulli:       y_k ~ Bernoulli( sigmoid(w_k^T x) )
    - Ordered probit:  y_k ~ OrderedProbit( w_k^T x, thresholds )
    """

    def __init__(self, params: DLSParams):
        self.params = params

    # ------------------------------------------------------------------
    # Log-likelihood contributions
    # ------------------------------------------------------------------
    def log_likelihood_counts(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Poisson log-likelihood for a batch.

        Parameters
        ----------
        x : np.ndarray, shape (n, 6)
        y : np.ndarray, shape (n, K_count)

        Returns
        -------
        ll : np.ndarray, shape (n,)
        """
        p = self.params
        eta = x @ p.w_count.T                    # (n, K)
        lam = np.exp(np.clip(eta, -10, 10))      # rate
        # Poisson log pmf
        ll = y * np.log(lam + 1e-12) - lam - special.gammaln(y + 1)
        return ll.sum(axis=1)

    def log_likelihood_binary(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Bernoulli log-likelihood for a batch."""
        p = self.params
        eta = x @ p.w_binary.T
        prob = 1.0 / (1.0 + np.exp(-np.clip(eta, -50, 50)))
        prob = np.clip(prob, 1e-9, 1 - 1e-9)
        ll = y * np.log(prob) + (1 - y) * np.log(1 - prob)
        return ll.sum(axis=1)

    def log_likelihood_ordinal(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Ordered probit log-likelihood for a batch."""
        p = self.params
        eta = x @ p.w_ordinal.T                       # (n, K)
        thresholds = p.ordinal_thresholds              # (M-1,)
        # Standardize eta
        # For each observation y in {0, ..., M-1}:
        #   P(y=0)  = Phi(t_0 - eta)
        #   P(y=m)  = Phi(t_m - eta) - Phi(t_{m-1} - eta)
        #   P(y=M-1)= 1 - Phi(t_{M-1} - eta)
        K = eta.shape[1]
        n = eta.shape[0]
        ll = np.zeros(n)
        for k in range(K):
            e = eta[:, k]
            yk = y[:, k].astype(int)
            # Build cumulative probabilities
            t = thresholds
            cdf = special.ndtr(t[None, :] - e[:, None])   # (n, M-1)
            zeros = np.zeros((n, 1))
            ones = np.ones((n, 1))
            cum = np.concatenate([zeros, cdf, ones], axis=1)  # (n, M)
            pk = cum[np.arange(n), yk + 1] - cum[np.arange(n), yk]
            ll += np.log(np.clip(pk, 1e-12, None))
        return ll

    # ------------------------------------------------------------------
    # Batch dispatch
    # ------------------------------------------------------------------
    def log_likelihood(
        self,
        x: np.ndarray,
        counts: Optional[np.ndarray] = None,
        binaries: Optional[np.ndarray] = None,
        ordinals: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Sum of log-likelihood contributions from all channels."""
        x = np.atleast_2d(x)
        ll = np.zeros(x.shape[0])
        if counts is not None:
            ll = ll + self.log_likelihood_counts(x, counts)
        if binaries is not None:
            ll = ll + self.log_likelihood_binary(x, binaries)
        if ordinals is not None:
            ll = ll + self.log_likelihood_ordinal(x, ordinals)
        return ll

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def sample(
        self,
        x: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ) -> Observation:
        """Draw observations for a batch of states."""
        rng = rng or np.random.default_rng()
        p = self.params
        x = np.atleast_2d(x)

        counts = None
        binaries = None
        ordinals = None

        if p.w_count is not None:
            eta = x @ p.w_count.T
            lam = np.exp(np.clip(eta, -10, 10))
            counts = rng.poisson(lam)

        if p.w_binary is not None:
            eta = x @ p.w_binary.T
            prob = 1.0 / (1.0 + np.exp(-np.clip(eta, -50, 50)))
            binaries = (rng.random(prob.shape) < prob).astype(int)

        if p.w_ordinal is not None:
            eta = x @ p.w_ordinal.T
            thresholds = p.ordinal_thresholds
            # Add standard normal noise to eta and compare with thresholds
            z = eta + rng.normal(0.0, 1.0, size=eta.shape)
            # Count how many thresholds z exceeds
            ordinals = (z[:, :, None] > thresholds[None, None, :]).sum(axis=2)
            ordinals = ordinals.astype(int)

        return Observation(counts=counts, binaries=binaries, ordinals=ordinals)