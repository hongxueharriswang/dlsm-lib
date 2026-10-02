"""Synthetic data generation for the DLSM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from dlsm.models.dlsm import DLSM
from dlsm.core.observations import Observation


@dataclass
class SimulatedData:
    """Container for a synthetic dataset."""

    states: np.ndarray            # (n_students, T, 6)
    statuses: np.ndarray          # (n_students, T)
    observations: List[List[Observation]]   # [student][time]
    events: np.ndarray            # (n_students, T, 4)


class Simulator:
    """Generate synthetic DLSM data.

    Parameters
    ----------
    model : DLSM
    seed : int, optional
    """

    def __init__(self, model: DLSM, seed: Optional[int] = None):
        self.model = model
        self.rng = np.random.default_rng(seed)

    def simulate(
        self,
        n_students: int = 50,
        n_timesteps: int = 60,
        event_prob: float = 0.3,
        x0: Optional[np.ndarray] = None,
    ) -> SimulatedData:
        """Simulate a cohort of students."""
        states = np.zeros((n_students, n_timesteps, 6))
        statuses = np.zeros((n_students, n_timesteps), dtype=int)
        events = np.zeros((n_students, n_timesteps, 4))
        observations: List[List[Observation]] = []

        for i in range(n_students):
            ev = (self.rng.random((n_timesteps, 4)) < event_prob).astype(int)
            s, st, obs = self.model.simulate_trajectory(
                n_timesteps=n_timesteps,
                x0=x0,
                events=ev,
                rng=self.rng,
            )
            states[i] = s
            statuses[i] = st
            events[i] = ev
            observations.append(obs)

        return SimulatedData(
            states=states,
            statuses=statuses,
            observations=observations,
            events=events,
        )