from __future__ import annotations

import numpy as np

from .reduce import reduce_losses
from .types import Reduction


def cross_entropy_from_probs(
    probs: np.ndarray,
    target: np.ndarray,
    reduction: Reduction = "mean",
    eps: float = 1e-12,
) -> np.ndarray | float:
    """
    Categorical cross-entropy from probability input.

    Supports:
    - class indices, shape (N,)
    - one-hot / soft target, shape (N, C)
    """
    if probs.ndim != 2:
        raise ValueError(f"probs must be 2D (N, C), got shape {probs.shape}")

    p = np.clip(probs, eps, 1.0 - eps)
    if target.ndim == 1:
        n = probs.shape[0]
        indices = target.astype(np.int64)
        losses = -np.log(p[np.arange(n), indices])
    elif target.ndim == 2 and target.shape == probs.shape:
        losses = -np.sum(target * np.log(p), axis=1)
    else:
        raise ValueError("target must be (N,) indices or (N, C) soft/one-hot")

    return reduce_losses(losses, reduction)


__all__ = ["cross_entropy_from_probs"]
