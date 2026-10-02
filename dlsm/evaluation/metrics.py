"""Evaluation metrics for filtering and prediction."""

from __future__ import annotations

from typing import Optional

import numpy as np


def rmse(true: np.ndarray, pred: np.ndarray, axis: int = 0) -> np.ndarray:
    """Root mean squared error along ``axis``."""
    return np.sqrt(np.mean((true - pred) ** 2, axis=axis))


def coverage(
    true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Proportion of true values inside [lower, upper]."""
    return float(np.mean((true >= lower) & (true <= upper)))


def mae(true: np.ndarray, pred: np.ndarray, axis: int = 0) -> np.ndarray:
    return np.mean(np.abs(true - pred), axis=axis)


def bias(true: np.ndarray, pred: np.ndarray, axis: int = 0) -> np.ndarray:
    return np.mean(pred - true, axis=axis)


def brier_score(prob: np.ndarray, outcome: np.ndarray) -> float:
    """Brier score for binary predictions."""
    return float(np.mean((prob - outcome) ** 2))


def auc(prob: np.ndarray, outcome: np.ndarray) -> float:
    """Area under the ROC curve (Mann–Whitney U statistic)."""
    prob = np.asarray(prob)
    outcome = np.asarray(outcome).astype(int)
    pos = prob[outcome == 1]
    neg = prob[outcome == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # Rank-based AUC
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg])))
    n_pos = len(pos)
    n_neg = len(neg)
    u = ranks[:n_pos].sum() - n_pos * (n_pos - 1) / 2
    return float(u / (n_pos * n_neg))


def parameter_recovery_table(
    true_params: np.ndarray,
    posterior_samples: np.ndarray,
    param_names: list[str],
    level: float = 0.95,
) -> list[dict]:
    """Produce a per-parameter recovery table."""
    lo = (1 - level) / 2
    hi = 1 - lo
    rows = []
    for j, name in enumerate(param_names):
        samples = posterior_samples[:, j]
        ci = np.quantile(samples, [lo, hi])
        rows.append({
            "parameter": name,
            "true": float(true_params[j]),
            "posterior_mean": float(samples.mean()),
            "posterior_std": float(samples.std()),
            "ci_lower": float(ci[0]),
            "ci_upper": float(ci[1]),
            "covered": bool(ci[0] <= true_params[j] <= ci[1]),
        })
    return rows