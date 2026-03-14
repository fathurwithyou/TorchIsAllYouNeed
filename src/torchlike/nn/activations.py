from __future__ import annotations

import torchlike.functional as F
from torchlike.tensor import Tensor

from .module import Module


class Identity(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.relu(x)


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.sigmoid(x)


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.tanh(x)


class Softmax(Module):
    def __init__(self, axis: int = -1) -> None:
        self.axis = axis

    def forward(self, x: Tensor) -> Tensor:
        return F.softmax(x, axis=self.axis)


class GELU(Module):
    def __init__(self, alpha: float = 1.702) -> None:
        self.alpha = alpha

    def forward(self, x: Tensor) -> Tensor:
        return F.gelu(x, alpha=self.alpha)


class SELU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.selu(x)
