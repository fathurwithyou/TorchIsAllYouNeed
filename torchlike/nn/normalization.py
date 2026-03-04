from __future__ import annotations

import numpy as np

from torchlike.tensor import Tensor

from .module import Module


class RMSNorm(Module):
    def __init__(
        self,
        normalized_shape: int,
        *,
        eps: float = 1e-8,
        affine: bool = True,
    ) -> None:
        if normalized_shape <= 0:
            raise ValueError("normalized_shape must be positive")

        self.normalized_shape = int(normalized_shape)
        self.eps = float(eps)
        self.affine = bool(affine)
        self.weight = (
            Tensor(np.ones((1, self.normalized_shape)), requires_grad=True)
            if self.affine
            else None
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.normalized_shape:
            raise ValueError(
                f"Expected last dimension {self.normalized_shape}, got {x.shape[-1]}"
            )

        rms = ((x * x).mean(axis=-1, keepdims=True) + self.eps) ** 0.5
        out = x / rms
        if self.weight is not None:
            out = out * self.weight
        return out
