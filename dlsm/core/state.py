"""State representation for the DLSM."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import numpy as np

from dlsm.core.parameters import (
    N_ENGAGEMENT,
    N_CONTINUOUS,
    N_STATE,
    N_STATUS,
)


class Status(IntEnum):
    REGISTERED = 0
    PROGRESSING = 1
    STOPPED_OUT = 2
    WITHDRAWN = 3
    COMPLETED = 4


@dataclass
class DLSMState:
    """A single state sample.

    Attributes
    ----------
    status : int
        Discrete status code (0-4).
    continuous : np.ndarray
        Continuous state of length 6:
        [e^O, e^P, e^C, e^SR, P, R].
    """

    status: int
    continuous: np.ndarray

    def __post_init__(self) -> None:
        self.continuous = np.asarray(self.continuous, dtype=float)
        if self.continuous.shape != (N_CONTINUOUS,):
            raise ValueError(
                f"continuous state must have length {N_CONTINUOUS}, "
                f"got {self.continuous.shape}"
            )
        if not (0 <= self.status < N_STATUS):
            raise ValueError(f"status must be in [0, {N_STATUS}), got {self.status}")

    # ---- convenience accessors -------------------------------------------
    @property
    def engagement(self) -> np.ndarray:
        return self.continuous[:N_ENGAGEMENT]

    @property
    def progress(self) -> float:
        return float(self.continuous[4])

    @property
    def risk(self) -> float:
        return float(self.continuous[5])

    @property
    def e_O(self) -> float:
        return float(self.continuous[0])

    @property
    def e_P(self) -> float:
        return float(self.continuous[1])

    @property
    def e_C(self) -> float:
        return float(self.continuous[2])

    @property
    def e_SR(self) -> float:
        return float(self.continuous[3])


def state_vector_to_array(states: np.ndarray) -> np.ndarray:
    """Stack a batch of continuous states into (n, 6)."""
    return np.asarray(states, dtype=float)