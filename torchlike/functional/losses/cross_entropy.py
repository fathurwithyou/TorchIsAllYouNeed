from __future__ import annotations

import numpy as np

from .reduce import reduce_losses
from .types import Reduction


def cross_entropy(
    logits: np.ndarray, target: np.ndarray, reduction: Reduction = "mean"
) -> np.ndarray | float:
    """
    Multi-class cross-entropy.

    Supports:
    - class indices, shape (N,)
    - soft targets / one-hot targets, shape (N, C)
    """
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2D (N, C), got shape {logits.shape}")

    shifted = logits - np.max(logits, axis=1, keepdims=True)
    logsumexp = np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
    log_probs = shifted - logsumexp

    if target.ndim == 1:
        n = logits.shape[0]
        indices = target.astype(np.int64)
        losses = -log_probs[np.arange(n), indices]
    elif target.ndim == 2 and target.shape == logits.shape:
        losses = -np.sum(target * log_probs, axis=1)
    else:
        raise ValueError(
            "target must have shape (N,) for class indices or (N, C) for soft/one-hot targets"
        )

    return reduce_losses(losses, reduction)


__all__ = ["cross_entropy"]
