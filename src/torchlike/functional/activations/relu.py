from __future__ import annotations

import numpy as np

from torchlike.tensor import Tensor


def relu(x: np.ndarray | Tensor) -> np.ndarray | Tensor:
    if isinstance(x, Tensor):
        return x.relu()
    return np.maximum(0.0, x)
