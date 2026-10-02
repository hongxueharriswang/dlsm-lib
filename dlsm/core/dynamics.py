"""Transition dynamics of the DLSM."""

from __future__ import annotations

from typing import Optional

import numpy as np

from dlsm.core.parameters import DLSParams, N_CONTINUOUS, N_STATUS
from dlsm.core.constraints import project_ordered_cone


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


class Dynamics:
    """Encapsulates the drift, noise, and status transition of the DLSM."""

    def __init__(self, params: DLSParams):
        self.params = params

    # ------------------------------------------------------------------
    # Continuous drift
    # ------------------------------------------------------------------
    def drift(
        self,
        x: np.ndarray,
        events: np.ndarray,
        dt: Optional[float] = None,
    ) -> np.ndarray:
        """Compute the drift f(x, events) for a batch of states.

        Parameters
        ----------
        x : np.ndarray
            Shape (n, 6) or (6,). Continuous state.
        events : np.ndarray
            Shape (n, 4) or (4,) binary indicators [a, p, c, s].
        dt : float, optional
            Time step. Defaults to params.dt.

        Returns
        -------
        np.ndarray
            Same shape as ``x``; the deterministic drift.
        """
        p = self.params
        dt = dt if dt is not None else p.dt

        x = np.atleast_2d(x)
        events = np.atleast_2d(events)
        n = x.shape[0]

        eO = x[:, 0]
        eP = x[:, 1]
        eC = x[:, 2]
        eSR = x[:, 3]
        P = x[:, 4]
        R = x[:, 5]

        a = events[:, 0]
        pp = events[:, 1]
        c = events[:, 2]
        s = events[:, 3]

        # Engagement drift
        d_eO = p.alpha_O * (1 - eO) * a - p.lambda_O * eO
        d_eP = p.alpha_P * eO * (1 - eP) * pp - p.lambda_P * eP
        d_eC = p.alpha_C * eP * (1 - eC) * c - p.lambda_C * eC
        d_eSR = p.alpha_SR * (1 - eSR) * s - p.lambda_SR * eSR

        # Progress drift (assessment completions handled separately)
        # Here we model only the "leakage" / noise part; progress jumps are
        # applied explicitly in ``step``.
        d_P = np.zeros(n)

        # Risk drift
        Ebar = (eO + eP + eC + eSR) / 4.0
        R_star = _sigmoid(
            p.omega_0
            - p.omega_1 * Ebar
            - p.omega_2 * P
            + p.omega_3 * 0.0  # inactivity passed via events? kept 0 here
        )
        d_R = p.gamma * (R_star - R)

        drift = np.stack([d_eO, d_eP, d_eC, d_eSR, d_P, d_R], axis=1) * dt
        return drift

    # ------------------------------------------------------------------
    # Progress jump from assessment completion
    # ------------------------------------------------------------------
    def progress_jump(
        self,
        x: np.ndarray,
        score: float,
        assessment_index: int = 0,
    ) -> np.ndarray:
        """Return the progress increment for a completed assessment."""
        p = self.params
        x = np.atleast_2d(x)
        eC = x[:, 2]
        eSR = x[:, 3]
        w_a = float(p.assessment_weights[
            min(assessment_index, len(p.assessment_weights) - 1)
        ])
        prob = _sigmoid(
            p.beta_0 + p.beta_1 * eC + p.beta_2 * eSR + p.beta_3 * score
        )
        return w_a * prob

    # ------------------------------------------------------------------
    # Full Euler-Maruyama step
    # ------------------------------------------------------------------
    def step(
        self,
        x: np.ndarray,
        events: np.ndarray,
        dt: Optional[float] = None,
        assessment_score: Optional[float] = None,
        assessment_index: int = 0,
        rng: Optional[np.random.Generator] = None,
        add_noise: bool = True,
    ) -> np.ndarray:
        """Advance the continuous state one step."""
        p = self.params
        dt = dt if dt is not None else p.dt
        rng = rng or np.random.default_rng()

        x = np.atleast_2d(x)
        events = np.atleast_2d(events)

        drift = self.drift(x, events, dt=dt)
        x_next = x + drift

        # Progress jump
        if assessment_score is not None:
            jump = self.progress_jump(x, assessment_score, assessment_index)
            x_next[:, 4] = x_next[:, 4] + jump

        # Process noise
        if add_noise:
            std = np.sqrt(p.q_diag) * np.sqrt(dt)
            noise = rng.normal(0.0, 1.0, size=x_next.shape) * std
            x_next = x_next + noise

        # Clip to [0, 1]
        x_next = np.clip(x_next, 0.0, 1.0)

        # Enforce ordered cone on engagement dimensions
        eng = project_ordered_cone(x_next[:, :4])
        x_next[:, :4] = eng

        return x_next

    # ------------------------------------------------------------------
    # Status transition
    # ------------------------------------------------------------------
    def status_logits(self, x: np.ndarray, status: np.ndarray) -> np.ndarray:
        """Compute logits for all next-status choices.

        Parameters
        ----------
        x : np.ndarray, shape (n, 6)
        status : np.ndarray, shape (n,), int

        Returns
        -------
        logits : np.ndarray, shape (n, N_STATUS)
        """
        p = self.params
        x = np.atleast_2d(x)
        status = np.atleast_1d(status).astype(int)
        n = x.shape[0]

        logits = np.zeros((n, N_STATUS))
        for i in range(n):
            s = status[i]
            # features: engagement (4) + progress (1) + risk (1)
            feats = x[i]
            logits[i] = p.w_status[s] @ feats
        return logits

    def status_transition_probs(
        self, x: np.ndarray, status: np.ndarray
    ) -> np.ndarray:
        """Return transition probability matrix for each sample."""
        logits = self.status_logits(x, status)
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)

    def sample_status(
        self,
        x: np.ndarray,
        status: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Sample the next status for a batch."""
        rng = rng or np.random.default_rng()
        probs = self.status_transition_probs(x, status)
        n = probs.shape[0]
        u = rng.random(n)
        cum = np.cumsum(probs, axis=1)
        new_status = (u[:, None] < cum).argmax(axis=1)
        return new_status.astype(int)