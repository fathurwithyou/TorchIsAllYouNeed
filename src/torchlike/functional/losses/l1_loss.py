from __future__ import annotations

import numpy as np

from .reduce import reduce_losses
from .types import Reduction


def l1_loss(
    pred: np.ndarray, target: np.ndarray, reduction: Reduction = "mean"
) -> np.ndarray | float:
    """L1 loss (absolute error)."""
    losses = np.abs(pred - target)
    return reduce_losses(losses, reduction)


__all__ = ["l1_loss"]
