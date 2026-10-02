"""Projection onto the ordered cone e^O >= e^P >= e^C."""

from __future__ import annotations

import numpy as np


def project_ordered_cone(e: np.ndarray) -> np.ndarray:
    """Euclidean projection of a batch of engagement vectors onto

        { e in [0,1]^4 : e[0] >= e[1] >= e[2] }

    The fourth component (self-regulated) is unconstrained except for [0,1].

    Parameters
    ----------
    e : np.ndarray
        Shape (..., 4). The last axis holds (e^O, e^P, e^C, e^SR).

    Returns
    -------
    np.ndarray
        Projected array with the same shape.
    """
    e = np.asarray(e, dtype=float)
    original_shape = e.shape
    e2 = e.reshape(-1, 4).copy()

    # Clip to [0, 1]
    e2 = np.clip(e2, 0.0, 1.0)

    # PAVA (pool adjacent violators) on the first three components
    x = e2[:, :3]
    n = x.shape[0]
    for i in range(n):
        x[i] = _pava_decreasing(x[i])

    e2[:, :3] = x
    return e2.reshape(original_shape)


def _pava_decreasing(y: np.ndarray) -> np.ndarray:
    """Pool Adjacent Violators Algorithm for a decreasing sequence.

    Finds the closest (in L2) non-increasing vector to y.
    """
    y = np.asarray(y, dtype=float).copy()
    n = len(y)
    if n == 0:
        return y

    # Work with negated values to use standard increasing PAVA
    z = -y
    # Standard PAVA for increasing sequence
    w = np.ones(n)
    block_start = 0
    block_end = 0
    values = np.empty(n)
    weights = np.empty(n)
    idx = 0

    i = 0
    while i < n:
        j = i
        # Extend the current block while it violates monotonicity
        while j + 1 < n and z[j + 1] < z[i]:
            j += 1
        # Average the block
        mean = z[i:j + 1].mean()
        for k in range(i, j + 1):
            z[k] = mean
        i = j + 1

    return -z