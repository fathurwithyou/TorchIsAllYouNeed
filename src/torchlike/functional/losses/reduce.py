from __future__ import annotations

import numpy as np

from .types import Reduction


def reduce_losses(losses: np.ndarray, reduction: Reduction) -> np.ndarray | float:
    if reduction == "none":
        return losses
    if reduction == "sum":
        return float(np.sum(losses))
    if reduction == "mean":
        return float(np.mean(losses))
    raise ValueError(f"Unsupported reduction: {reduction}")


__all__ = ["reduce_losses"]
