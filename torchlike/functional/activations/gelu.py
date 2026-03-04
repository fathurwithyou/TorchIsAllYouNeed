from __future__ import annotations

import numpy as np

from torchlike.tensor import Tensor


def gelu(x: np.ndarray | Tensor, alpha: float = 1.702) -> np.ndarray | Tensor:
    alpha = float(alpha)
    if isinstance(x, Tensor):
        return x * (x * alpha).sigmoid()
    return x * (1.0 / (1.0 + np.exp(-(alpha * x))))
