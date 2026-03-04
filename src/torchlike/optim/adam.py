from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from torchlike.tensor import Tensor
from .base import Optimizer


class Adam(Optimizer):
    def __init__(
        self,
        params: Iterable[Tensor],
        *,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        l1_lambda: float = 0.0,
        l2_lambda: float = 0.0,
    ) -> None:
        defaults = {
            "lr": float(lr),
            "beta1": float(beta1),
            "beta2": float(beta2),
            "eps": float(eps),
            "l1_lambda": float(l1_lambda),
            "l2_lambda": float(l2_lambda),
        }

        if not (0.0 <= defaults["beta1"] < 1.0 and 0.0 <= defaults["beta2"] < 1.0):
            raise ValueError("beta1 and beta2 must be in [0, 1)")
        if defaults["lr"] <= 0.0:
            raise ValueError("lr must be positive")
        if defaults["eps"] <= 0.0:
            raise ValueError("eps must be positive")

        super().__init__(params, defaults)
        self.t = 0

    def step(self) -> None:
        self.t += 1
        for group in self.param_groups:
            lr = float(group["lr"])
            beta1 = float(group["beta1"])
            beta2 = float(group["beta2"])
            eps = float(group["eps"])
            l1_lambda = float(group["l1_lambda"])
            l2_lambda = float(group["l2_lambda"])

            bias_correction1 = 1.0 - beta1**self.t
            bias_correction2 = 1.0 - beta2**self.t

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state.setdefault(p, {})
                if "m" not in state:
                    state["m"] = np.zeros_like(p.data)
                    state["v"] = np.zeros_like(p.data)

                reg_grad = 0.0
                if l1_lambda != 0.0:
                    reg_grad = reg_grad + l1_lambda * np.sign(p.data)
                if l2_lambda != 0.0:
                    reg_grad = reg_grad + l2_lambda * p.data

                grad = p.grad + reg_grad
                state["m"] = beta1 * state["m"] + (1.0 - beta1) * grad
                state["v"] = beta2 * state["v"] + (1.0 - beta2) * (grad * grad)

                m_hat = state["m"] / bias_correction1
                v_hat = state["v"] / bias_correction2
                p.data -= lr * (m_hat / (np.sqrt(v_hat) + eps))
