"""Main DLSM class tying together dynamics, observations, and constraints."""

from __future__ import annotations

from typing import Optional

import numpy as np

from dlsm.core.parameters import DLSParams
from dlsm.core.dynamics import Dynamics
from dlsm.core.observations import ObservationModel, Observation
from dlsm.core.constraints import project_ordered_cone
from dlsm.core.state import Status


class DLSM:
    """Dynamic Latent-State Model of student engagement.

    Parameters
    ----------
    params : DLSParams
        Model parameters.
    """

    def __init__(self, params: Optional[DLSParams] = None):
        self.params = params or DLSParams.default()
        self.dynamics = Dynamics(self.params)
        self.observation = ObservationModel(self.params)

    # ------------------------------------------------------------------
    # Forward simulation (single trajectory)
    # ------------------------------------------------------------------
    def simulate_trajectory(
        self,
        n_timesteps: int,
        x0: Optional[np.ndarray] = None,
        status0: int = Status.REGISTERED,
        events: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> tuple[np.ndarray, np.ndarray, list[Observation]]:
        """Simulate a single student trajectory.

        Returns
        -------
        states : (T, 6) continuous states
        statuses : (T,) discrete statuses
        observations : list of T Observation objects
        """
        rng = rng or np.random.default_rng()
        x0 = x0 if x0 is not None else np.array([0.1, 0.05, 0.01, 0.2, 0.0, 0.8])

        states = np.zeros((n_timesteps, 6))
        statuses = np.zeros(n_timesteps, dtype=int)
        obs_list: list[Observation] = []

        x = x0.copy()
        s = int(status0)
        states[0] = x
        statuses[0] = s

        for t in range(n_timesteps - 1):
            ev = events[t] if events is not None else rng.integers(0, 2, size=4)
            x = self.dynamics.step(x[None, :], ev[None, :], rng=rng)[0]
            s = int(self.dynamics.sample_status(x[None, :], np.array([s]), rng=rng)[0])
            states[t + 1] = x
            statuses[t + 1] = s

        for t in range(n_timesteps):
            obs = self.observation.sample(states[t][None, :], rng=rng)
            obs_list.append(obs)

        return states, statuses, obs_list

    # ------------------------------------------------------------------
    # Log-likelihood for a single observation
    # ------------------------------------------------------------------
    def log_likelihood(
        self,
        x: np.ndarray,
        observation: Observation,
    ) -> np.ndarray:
        """Log-likelihood of an observation given a batch of states."""
        counts = observation.counts[None, :] if observation.counts is not None else None
        binaries = observation.binaries[None, :] if observation.binaries is not None else None
        ordinals = observation.ordinals[None, :] if observation.ordinals is not None else None
        return self.observation.log_likelihood(x, counts, binaries, ordinals)