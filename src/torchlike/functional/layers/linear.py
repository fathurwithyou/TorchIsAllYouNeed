from __future__ import annotations

import numpy as np

from torchlike.tensor import Tensor


def linear(
    x: np.ndarray | Tensor,
    weight: np.ndarray | Tensor,
    bias: np.ndarray | Tensor | None = None,
) -> np.ndarray | Tensor:
    """Apply an affine transform: y = x @ weight + bias."""
    if isinstance(x, Tensor) or isinstance(weight, Tensor) or isinstance(bias, Tensor):
        x_t = x if isinstance(x, Tensor) else Tensor(x, requires_grad=False)
        w_t = (
            weight
            if isinstance(weight, Tensor)
            else Tensor(weight, requires_grad=False)
        )
        output = x_t @ w_t
        if bias is not None:
            b_t = (
                bias if isinstance(bias, Tensor) else Tensor(bias, requires_grad=False)
            )
            output = output + b_t
        return output

    output = x @ weight
    if bias is not None:
        output = output + bias
    return output
