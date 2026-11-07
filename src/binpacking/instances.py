"""
Input sequences, frequency predictions and the lower bound on OPT.

Sizes are integers in 1..k with bin capacity k (k = 100 by default), so a
distribution over sizes is just a probability vector of length k + 1 (index 0
unused).
"""
from __future__ import annotations

import numpy as np


def weibull_dist(k: int = 100, shape: float = 3.0, scale: float = 45.0, n_samples: int = 2_000_000, seed: int = 0):
    """Size distribution of round(scale * Weibull(shape)), clipped to 1..k.

    Weibull item sizes are a standard benchmark for bin packing; shape 3
    gives a bell-shaped spread of mid-sized items.
    """
    rng = np.random.default_rng(seed)
    x = np.clip(np.rint(scale * rng.weibull(shape, n_samples)), 1, k).astype(int)
    p = np.bincount(x, minlength=k + 1).astype(float)
    return p / p.sum()


def sample_items(dist: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.choice(len(dist), size=n, p=dist)


def true_frequencies(items, k: int = 100) -> np.ndarray:
    f = np.bincount(np.asarray(items), minlength=k + 1).astype(float)
    return f / f.sum()


def predicted_frequencies(true_dist: np.ndarray, wrong_dist: np.ndarray, delta: float) -> np.ndarray:
    """Prediction that is a (1 - delta, delta) mix of the truth and a wrong distribution.

    delta = 0 is a perfect prediction; delta = 1 is entirely wrong.
    """
    return (1 - delta) * true_dist + delta * wrong_dist


def l1_error(pred: np.ndarray, actual: np.ndarray) -> float:
    """Prediction error eta = sum_x |f'_x - f_x| (between 0 and 2)."""
    return float(np.abs(pred - actual).sum())


def lower_bound(items, k: int = 100) -> int:
    """No packing can use fewer than ceil(total size / k) bins."""
    return int(-(-int(np.sum(items)) // k))
