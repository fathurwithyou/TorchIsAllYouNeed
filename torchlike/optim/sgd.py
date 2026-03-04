from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from torchlike.tensor import Tensor
from .base import Optimizer


class SGD(Optimizer):
    def __init__(
        self,
        params: Iterable[Tensor],
        *,
        lr: float = 1e-2,
        l1_lambda: float = 0.0,
        l2_lambda: float = 0.0,
    ) -> None:
        defaults = {
            "lr": float(lr),
            "l1_lambda": float(l1_lambda),
            "l2_lambda": float(l2_lambda),
        }
        super().__init__(params, defaults)

    def step(self) -> None:
        for group in self.param_groups:
            lr = float(group["lr"])
            l1_lambda = float(group["l1_lambda"])
            l2_lambda = float(group["l2_lambda"])
            for p in group["params"]:
                if p.grad is None:
                    continue
                reg_grad = 0.0
                if l1_lambda != 0.0:
                    reg_grad = reg_grad + l1_lambda * np.sign(p.data)
                if l2_lambda != 0.0:
                    reg_grad = reg_grad + l2_lambda * p.data
                p.data -= lr * (p.grad + reg_grad)
