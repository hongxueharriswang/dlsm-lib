"""Particle Marginal Metropolis–Hastings for parameter estimation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
from tqdm import tqdm

from dlsm.models.dlsm import DLSM
from dlsm.core.parameters import DLSParams
from dlsm.core.observations import Observation
from dlsm.inference.particle_filter import ParticleFilter


@dataclass
class PMCMCResult:
    """Output of a PMCMC run."""

    chain: np.ndarray                  # (n_iter, n_params)
    log_posterior: np.ndarray          # (n_iter,)
    acceptance_rate: float
    param_names: List[str] = field(default_factory=list)

    def posterior_mean(self) -> np.ndarray:
        return self.chain.mean(axis=0)

    def posterior_std(self) -> np.ndarray:
        return self.chain.std(axis=0)

    def credible_interval(self, level: float = 0.95) -> np.ndarray:
        lo = (1 - level) / 2
        hi = 1 - lo
        return np.quantile(self.chain, [lo, hi], axis=0)


class PMCMC:
    """Particle Marginal Metropolis–Hastings sampler.

    Parameters
    ----------
    model : DLSM
    n_particles : int
        Number of particles used inside the likelihood estimate.
    n_iterations : int
        Number of MCMC iterations.
    burn_in : int
        Number of initial iterations to discard.
    thin : int
        Keep every ``thin``-th sample after burn-in.
    proposal_scale : float or np.ndarray
        Standard deviation of the Gaussian random-walk proposal.
    seed : int, optional
    """

    PARAM_NAMES = [
        "alpha_O", "alpha_P", "alpha_C", "alpha_SR",
        "lambda_O", "lambda_P", "lambda_C", "lambda_SR",
        "beta_0", "beta_1", "beta_2", "beta_3",
        "gamma",
        "omega_0", "omega_1", "omega_2", "omega_3",
    ]

    def __init__(
        self,
        model: DLSM,
        n_particles: int = 2000,
        n_iterations: int = 10000,
        burn_in: int = 2000,
        thin: int = 10,
        proposal_scale: float | np.ndarray = 0.02,
        seed: Optional[int] = None,
    ):
        self.model = model
        self.n_particles = int(n_particles)
        self.n_iterations = int(n_iterations)
        self.burn_in = int(burn_in)
        self.thin = int(thin)
        self.rng = np.random.default_rng(seed)

        n_params = len(self.PARAM_NAMES)
        if np.isscalar(proposal_scale):
            self.proposal_scale = np.full(n_params, float(proposal_scale))
        else:
            self.proposal_scale = np.asarray(proposal_scale, dtype=float)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(
        self,
        observations: List[Observation],
        events: Optional[np.ndarray] = None,
        initial_params: Optional[DLSParams] = None,
    ) -> PMCMCResult:
        """Run PMCMC on a single student's trajectory."""
        initial_params = initial_params or self.model.params
        template = initial_params

        theta = initial_params.as_vector()
        log_post = self._log_posterior(theta, observations, events, template)

        n_keep = (self.n_iterations - self.burn_in) // self.thin
        chain = np.zeros((n_keep, len(theta)))
        log_post_chain = np.zeros(n_keep)

        n_accept = 0
        keep_idx = 0

        for it in tqdm(range(self.n_iterations), desc="PMCMC"):
            # Propose
            proposal = theta + self.rng.normal(
                0.0, self.proposal_scale, size=theta.shape
            )
            # Enforce positivity on rate parameters
            proposal = np.abs(proposal)

            log_post_prop = self._log_posterior(
                proposal, observations, events, template
            )

            # Accept/reject
            log_alpha = log_post_prop - log_post
            if np.log(self.rng.random()) < log_alpha:
                theta = proposal
                log_post = log_post_prop
                n_accept += 1

            if it >= self.burn_in and (it - self.burn_in) % self.thin == 0:
                if keep_idx < n_keep:
                    chain[keep_idx] = theta
                    log_post_chain[keep_idx] = log_post
                    keep_idx += 1

        return PMCMCResult(
            chain=chain[:keep_idx],
            log_posterior=log_post_chain[:keep_idx],
            acceptance_rate=n_accept / self.n_iterations,
            param_names=list(self.PARAM_NAMES),
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _log_posterior(
        self,
        theta: np.ndarray,
        observations: List[Observation],
        events: Optional[np.ndarray],
        template: DLSParams,
    ) -> float:
        params = DLSParams.from_vector(theta, template=template)
        model = DLSM(params)
        pf = ParticleFilter(
            model, n_particles=self.n_particles, seed=int(self.rng.integers(0, 1e9))
        )
        ll = pf.estimate_log_likelihood(observations, events=events)
        lp = self._log_prior(theta)
        return ll + lp

    def _log_prior(self, theta: np.ndarray) -> float:
        """Weakly informative prior: Gamma(2, 2) on positive parameters."""
        # All parameters in our vector are positive rates (except intercepts).
        # We use a soft positivity prior and a half-normal on the rest.
        lp = 0.0
        for i, name in enumerate(self.PARAM_NAMES):
            v = theta[i]
            if name in ("beta_0",):
                # Intercept: normal(0, 2)
                lp += -0.5 * (v / 2.0) ** 2
            else:
                # Positive rate: Gamma(2, 2)
                if v <= 0:
                    return -np.inf
                lp += (2 - 1) * np.log(v) - 2 * v
        return float(lp)