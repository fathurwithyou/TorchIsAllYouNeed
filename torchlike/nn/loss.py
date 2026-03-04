from __future__ import annotations

import numpy as np

from torchlike.tensor import Tensor

from .module import Module


def _ensure_tensor(x) -> Tensor:
    if isinstance(x, Tensor):
        return x
    return Tensor(x, requires_grad=False)


def _as_one_hot(target: np.ndarray, num_classes: int) -> np.ndarray:
    one_hot = np.zeros((target.shape[0], num_classes), dtype=np.float64)
    one_hot[np.arange(target.shape[0]), target.astype(np.int64)] = 1.0
    return one_hot


class _Loss(Module):
    reduction: str = "mean"

    def __init__(self, reduction: str = "mean") -> None:
        self.reduction = reduction
        if self.reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Invalid reduction: {self.reduction}")


class MSELoss(_Loss):
    def forward(self, pred, target) -> Tensor:
        p = _ensure_tensor(pred)
        t = _ensure_tensor(target)
        diff = p - t
        return (diff * diff).mean()


class BCELoss(_Loss):
    def __init__(self, eps: float = 1e-12) -> None:
        self.eps = eps

    def forward(self, pred, target) -> Tensor:
        p = _ensure_tensor(pred).clamp(self.eps, 1.0 - self.eps)
        t = _ensure_tensor(target)
        one = Tensor(1.0)
        return -(t * p.log() + (one - t) * (one - p).log()).mean()


class CrossEntropyLoss(_Loss):
    def __init__(self, eps: float = 1e-12) -> None:
        self.eps = eps

    def forward(self, pred, target) -> Tensor:
        p = _ensure_tensor(pred).clamp(self.eps, 1.0 - self.eps)

        if isinstance(target, Tensor):
            target_np = target.data
        else:
            target_np = np.asarray(target)

        if target_np.ndim == 1:
            one_hot = _as_one_hot(target_np, p.shape[1])
            t = Tensor(one_hot, requires_grad=False)
        elif target_np.ndim == 2 and target_np.shape == p.shape:
            t = Tensor(target_np, requires_grad=False)
        else:
            raise ValueError("target must be class indices (N,) or one-hot (N, C)")

        return -(t * p.log()).sum(axis=1).mean()


class BCEWithLogitsLoss(_Loss):
    def forward(self, pred, target) -> Tensor:
        p = _ensure_tensor(pred)
        t = _ensure_tensor(target)
        abs_p = p.relu() + (-p).relu()
        return (p.relu() - p * t + (Tensor(1.0) + (-abs_p).exp()).log()).mean()
