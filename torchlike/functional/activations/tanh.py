from __future__ import annotations

import numpy as np

from torchlike.tensor import Tensor


def tanh(x: np.ndarray | Tensor) -> np.ndarray | Tensor:
    if isinstance(x, Tensor):
        return x.tanh()
    return np.tanh(x)
