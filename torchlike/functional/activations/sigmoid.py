from __future__ import annotations

import numpy as np

from torchlike.tensor import Tensor


def sigmoid(x: np.ndarray | Tensor) -> np.ndarray | Tensor:
    if isinstance(x, Tensor):
        return x.sigmoid()
    clipped = np.clip(x, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-clipped))
