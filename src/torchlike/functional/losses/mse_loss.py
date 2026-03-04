from __future__ import annotations

import numpy as np

from .reduce import reduce_losses
from .types import Reduction


def mse_loss(
    pred: np.ndarray, target: np.ndarray, reduction: Reduction = "mean"
) -> np.ndarray | float:
    """Mean squared error loss."""
    losses = (pred - target) ** 2
    return reduce_losses(losses, reduction)


__all__ = ["mse_loss"]
