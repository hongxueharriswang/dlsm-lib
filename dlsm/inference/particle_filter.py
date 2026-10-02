"""Bootstrap particle filter with systematic resampling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from dlsm.models.dlsm import DLSM
from dlsm.core.observations import Observation


@dataclass
class FilterResult:
    """Output of a particle-filter pass."""

    mean_states: np.ndarray           # (T, 6)
    var_states: np.ndarray            # (T, 6)
    status_probs: np.ndarray          # (T, N_STATUS)
    mean_status: np.ndarray           # (T,)
    log_likelihood: float             # total log-likelihood (unbiased estimate)
    ess: np.ndarray                   # (T,) effective sample size
    particles: Optional[np.ndarray] = None   # (T, N, 6) if requested
    weights: Optional[np.ndarray] = None     # (T, N) if requested


class ParticleFilter:
    """Bootstrap particle filter for the DLSM.

    Parameters
    ----------
    model : DLSM
    n_particles : int
        Number of particles.
    resample_threshold : float
        Resample when ESS falls below this fraction of N.
    seed : int, optional
    """

    def __init__(
        self,
        model: DLSM,
        n_particles: int = 1000,
        resample_threshold: float = 0.5,
        seed: Optional[int] = None,
    ):
        self.model = model
        self.n_particles = int(n_particles)
        self.resample_threshold = float(resample_threshold)
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def filter(
        self,
        observations: List[Observation],
        events: Optional[np.ndarray] = None,
        x0: Optional[np.ndarray] = None,
        store_particles: bool = False,
    ) -> FilterResult:
        """Run the particle filter on a single student trajectory."""
        T = len(observations)
        N = self.n_particles
        p = self.model.params

        # Initial particles
        if x0 is None:
            x0 = np.array([0.1, 0.05, 0.01, 0.2, 0.0, 0.8])
        particles = np.tile(x0, (N, 1)) + self.rng.normal(
            0.0, 0.05, size=(N, 6)
        )
        particles = np.clip(particles, 0.0, 1.0)
        particles[:, :4] = _project_ordered_cone_batch(particles[:, :4])

        statuses = np.full(N, 0, dtype=int)   # start Registered
        log_w = np.full(N, -np.log(N))

        mean_states = np.zeros((T, 6))
        var_states = np.zeros((T, 6))
        status_probs = np.zeros((T, 5))
        mean_status = np.zeros(T, dtype=int)
        ess_arr = np.zeros(T)
        log_lik_total = 0.0

        particles_hist = np.zeros((T, N, 6)) if store_particles else None
        weights_hist = np.zeros((T, N)) if store_particles else None

        for t in range(T):
            obs = observations[t]

            # --- Predict ---
            ev = events[t] if events is not None else np.zeros(4)
            ev_batch = np.tile(ev, (N, 1))
            particles = self.model.dynamics.step(
                particles, ev_batch, rng=self.rng, add_noise=True
            )
            statuses = self.model.dynamics.sample_status(
                particles, statuses, rng=self.rng
            )

            # --- Update ---
            ll = self.model.log_likelihood(particles, obs)
            log_w = log_w + ll
            log_lik_total += _logsumexp(log_w) - np.log(N)

            # Normalize weights
            w = _normalize_log_weights(log_w)
            ess = 1.0 / np.sum(w ** 2)
            ess_arr[t] = ess

            # --- Summaries ---
            mean_states[t] = np.sum(w[:, None] * particles, axis=0)
            var_states[t] = np.sum(w[:, None] * (particles - mean_states[t]) ** 2, axis=0)
            for s in range(5):
                status_probs[t, s] = np.sum(w[statuses == s])
            mean_status[t] = int(np.argmax(status_probs[t]))

            if store_particles:
                particles_hist[t] = particles
                weights_hist[t] = w

            # --- Resample ---
            if ess < self.resample_threshold * N:
                idx = _systematic_resample(w, self.rng)
                particles = particles[idx]
                statuses = statuses[idx]
                log_w = np.full(N, -np.log(N))

        return FilterResult(
            mean_states=mean_states,
            var_states=var_states,
            status_probs=status_probs,
            mean_status=mean_status,
            log_likelihood=float(log_lik_total),
            ess=ess_arr,
            particles=particles_hist,
            weights=weights_hist,
        )

    # ------------------------------------------------------------------
    # Likelihood-only interface for PMCMC
    # ------------------------------------------------------------------
    def estimate_log_likelihood(
        self,
        observations: List[Observation],
        events: Optional[np.ndarray] = None,
        x0: Optional[np.ndarray] = None,
    ) -> float:
        """Return the particle-filter estimate of log p(y_{1:T})."""
        return self.filter(observations, events=events, x0=x0).log_likelihood


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _normalize_log_weights(log_w: np.ndarray) -> np.ndarray:
    m = np.max(log_w)
    w = np.exp(log_w - m)
    s = w.sum()
    if s <= 0 or not np.isfinite(s):
        return np.full_like(w, 1.0 / len(w))
    return w / s


def _logsumexp(log_w: np.ndarray) -> float:
    m = np.max(log_w)
    return float(m + np.log(np.sum(np.exp(log_w - m))))


def _systematic_resample(w: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    N = len(w)
    positions = (rng.random() + np.arange(N)) / N
    cumsum = np.cumsum(w)
    cumsum[-1] = 1.0
    idx = np.searchsorted(cumsum, positions)
    return np.clip(idx, 0, N - 1)


def _project_ordered_cone_batch(e: np.ndarray) -> np.ndarray:
    from dlsm.core.constraints import project_ordered_cone
    return project_ordered_cone(e)