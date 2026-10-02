"""Parameter containers for the Dynamic Latent-State Model."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple

import numpy as np


# Status labels used throughout the library
STATUS_LABELS = ("Registered", "Progressing", "Stopped-Out", "Withdrawn", "Completed")
N_STATUS = len(STATUS_LABELS)
N_ENGAGEMENT = 4          # e^O, e^P, e^C, e^SR
N_CONTINUOUS = N_ENGAGEMENT + 2   # + progress + risk
N_STATE = N_CONTINUOUS            # continuous part of the state (status is discrete)


@dataclass
class DLSParams:
    """Full parameter set of the DLSM.

    All continuous parameters are stored as numpy arrays for efficient
    vectorized operations inside the particle filter.
    """

    # --- Engagement dynamics (growth) ---
    alpha_O: float = 0.30
    alpha_P: float = 0.25
    alpha_C: float = 0.20
    alpha_SR: float = 0.35

    # --- Engagement dynamics (decay) ---
    lambda_O: float = 0.15
    lambda_P: float = 0.20
    lambda_C: float = 0.25
    lambda_SR: float = 0.10

    # --- Progress dynamics ---
    beta_0: float = -0.5
    beta_1: float = 0.8    # weight on e^C
    beta_2: float = 0.6    # weight on e^SR
    beta_3: float = 1.2    # weight on assessment score
    assessment_weights: np.ndarray = field(
        default_factory=lambda: np.array([1.0])
    )

    # --- Risk dynamics ---
    gamma: float = 0.20
    omega_0: float = 1.5
    omega_1: float = 2.0   # weight on mean engagement
    omega_2: float = 1.5   # weight on progress
    omega_3: float = 0.05  # weight on days inactive

    # --- Status transitions ---
    # w_s[s, s', :] is the coefficient vector for transition s -> s'
    # with features z = [e^O, e^P, e^C, e^SR, P, R] (6 features)
    w_status: np.ndarray = field(
        default_factory=lambda: _default_status_coefficients()
    )
    n_status_features: int = 6

    # --- Process noise covariance (diagonal, length 6) ---
    q_diag: np.ndarray = field(
        default_factory=lambda: np.array([0.02, 0.02, 0.02, 0.02, 0.01, 0.02])
    )

    # --- Observation noise (per channel) ---
    # Poisson channels: no noise parameter (variance = mean)
    # Bernoulli: no noise parameter
    # Ordered probit: thresholds
    ordinal_thresholds: np.ndarray = field(
        default_factory=lambda: np.array([-1.0, 0.0, 1.0])
    )

    # --- Observation weight matrices ---
    # Each row maps the 6-dim continuous state to a natural parameter.
    w_count: Optional[np.ndarray] = None      # shape (K_count, 6)
    w_binary: Optional[np.ndarray] = None     # shape (K_binary, 6)
    w_ordinal: Optional[np.ndarray] = None    # shape (K_ordinal, 6)

    # --- Discretization step ---
    dt: float = 1.0

    def __post_init__(self) -> None:
        # Ensure arrays are float64
        self.assessment_weights = np.asarray(self.assessment_weights, dtype=float)
        self.q_diag = np.asarray(self.q_diag, dtype=float).reshape(-1)
        self.ordinal_thresholds = np.asarray(self.ordinal_thresholds, dtype=float)
        self.w_status = np.asarray(self.w_status, dtype=float)

        if self.w_count is None:
            # Default: 4 count channels (one per engagement dimension)
            self.w_count = np.array([
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            ], dtype=float)
        else:
            self.w_count = np.asarray(self.w_count, dtype=float)

        if self.w_binary is None:
            # Default: 1 binary channel (assessment submission) driven by progress
            self.w_binary = np.array([[0.0, 0.0, 0.0, 0.0, 1.0, 0.0]], dtype=float)
        else:
            self.w_binary = np.asarray(self.w_binary, dtype=float)

        if self.w_ordinal is None:
            # Default: 1 ordinal channel driven by self-regulated engagement
            self.w_ordinal = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]], dtype=float)
        else:
            self.w_ordinal = np.asarray(self.w_ordinal, dtype=float)

        if self.q_diag.shape != (N_CONTINUOUS,):
            raise ValueError(
                f"q_diag must have length {N_CONTINUOUS}, got {self.q_diag.shape}"
            )
        if self.w_status.shape != (N_STATUS, N_STATUS, self.n_status_features):
            raise ValueError(
                f"w_status must have shape ({N_STATUS}, {N_STATUS}, "
                f"{self.n_status_features}), got {self.w_status.shape}"
            )

    # --- Convenience constructors -------------------------------------------
    @classmethod
    def default(cls) -> "DLSParams":
        """Return a sensible default parameter set."""
        return cls()

    def as_vector(self) -> np.ndarray:
        """Flatten all continuous parameters into a 1-D vector for PMCMC."""
        parts = [
            np.array([
                self.alpha_O, self.alpha_P, self.alpha_C, self.alpha_SR,
                self.lambda_O, self.lambda_P, self.lambda_C, self.lambda_SR,
                self.beta_0, self.beta_1, self.beta_2, self.beta_3,
                self.gamma,
                self.omega_0, self.omega_1, self.omega_2, self.omega_3,
            ])
        ]
        return np.concatenate(parts)

    @classmethod
    def from_vector(cls, vec: np.ndarray, template: Optional["DLSParams"] = None) -> "DLSParams":
        """Reconstruct a DLSParams from a flat vector."""
        template = template or cls.default()
        p = DLSParams(
            alpha_O=vec[0], alpha_P=vec[1], alpha_C=vec[2], alpha_SR=vec[3],
            lambda_O=vec[4], lambda_P=vec[5], lambda_C=vec[6], lambda_SR=vec[7],
            beta_0=vec[8], beta_1=vec[9], beta_2=vec[10], beta_3=vec[11],
            gamma=vec[12],
            omega_0=vec[13], omega_1=vec[14], omega_2=vec[15], omega_3=vec[16],
            assessment_weights=template.assessment_weights.copy(),
            w_status=template.w_status.copy(),
            q_diag=template.q_diag.copy(),
            ordinal_thresholds=template.ordinal_thresholds.copy(),
            w_count=template.w_count.copy(),
            w_binary=template.w_binary.copy(),
            w_ordinal=template.w_ordinal.copy(),
            dt=template.dt,
        )
        return p

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in list(d.items()):
            if isinstance(v, np.ndarray):
                d[k] = v.tolist()
        return d


def _default_status_coefficients() -> np.ndarray:
    """Return a default set of multinomial-logistic coefficients.

    Shape: (5, 5, 6). The first index is the *current* status, the second
    the *next* status, and the third the feature index. Absorbing states
    (Withdrawn, Completed) have large negative intercepts for any outgoing
    transition, and self-transitions have a large positive intercept.
    """
    rng = np.random.default_rng(0)
    w = rng.normal(0.0, 0.3, size=(N_STATUS, N_STATUS, 6))

    # Self-transition intercept is high
    for s in range(N_STATUS):
        w[s, s, 4] += 1.5   # progress feature boosts persistence

    # Withdrawn and Completed are absorbing (intercept -10 for outgoing)
    for s in (3, 4):
        for s2 in range(N_STATUS):
            if s2 != s:
                w[s, s2, :] = -10.0

    # Registered -> Progressing is likely
    w[0, 1, :] += np.array([0.0, 0.0, 0.0, 0.0, 2.0, 0.0])
    # Progressing -> Stopped-Out is driven by risk
    w[1, 2, :] += np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    # Stopped-Out -> Progressing is driven by engagement
    w[2, 1, :] += np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
    # Progressing -> Withdrawn is driven by risk
    w[1, 3, :] += np.array([0.0, 0.0, 0.0, 0.0, 0.0, 2.0])
    # Progressing -> Completed is driven by progress
    w[1, 4, :] += np.array([0.0, 0.0, 0.0, 0.0, 2.0, 0.0])
    return w