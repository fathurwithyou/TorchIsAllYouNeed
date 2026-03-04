from __future__ import annotations

import numpy as np

from torchlike.tensor import Tensor


SELU_ALPHA = 1.6732632423543772
SELU_LAMBDA = 1.0507009873554805


def selu(x: np.ndarray | Tensor) -> np.ndarray | Tensor:
    if isinstance(x, Tensor):
        pos = x.relu()
        neg = (-x).relu()
        elu = pos + SELU_ALPHA * ((-neg).exp() - 1.0)
        return SELU_LAMBDA * elu
    return SELU_LAMBDA * np.where(x > 0.0, x, SELU_ALPHA * (np.exp(x) - 1.0))
