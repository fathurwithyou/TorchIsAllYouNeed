from __future__ import annotations

import numpy as np

from .reduce import reduce_losses
from .types import Reduction


def binary_cross_entropy(
    pred: np.ndarray,
    target: np.ndarray,
    reduction: Reduction = "mean",
    eps: float = 1e-12,
) -> np.ndarray | float:
    """Binary cross-entropy on probabilities."""
    pred_clipped = np.clip(pred, eps, 1.0 - eps)
    losses = -(
        target * np.log(pred_clipped) + (1.0 - target) * np.log(1.0 - pred_clipped)
    )
    return reduce_losses(losses, reduction)


__all__ = ["binary_cross_entropy"]
