from __future__ import annotations

import numpy as np

from torchlike.tensor import Tensor


def softmax(x: np.ndarray | Tensor, axis: int = -1) -> np.ndarray | Tensor:
    if isinstance(x, Tensor):
        return x.softmax(axis=axis)
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=axis, keepdims=True)
