from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np


def _to_array(data: Any) -> np.ndarray:
    if isinstance(data, np.ndarray):
        return data.astype(np.float64)
    return np.array(data, dtype=np.float64)


def _unbroadcast(grad: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    # Collapse a broadcasted gradient back to the operand's original shape.
    if grad.shape == target_shape:
        return grad

    g = grad
    while len(g.shape) > len(target_shape):
        g = np.sum(g, axis=0)

    for axis, (g_dim, t_dim) in enumerate(zip(g.shape, target_shape)):
        if t_dim == 1 and g_dim != 1:
            g = np.sum(g, axis=axis, keepdims=True)

    return g.reshape(target_shape)


class Tensor:
    def __init__(
        self,
        data: Any,
        *,
        requires_grad: bool = False,
        _children: Iterable[Tensor] = (),
        _op: str = "",
    ) -> None:
        self.data = _to_array(data)
        self.requires_grad = requires_grad
        self.grad: np.ndarray | None = (
            np.zeros_like(self.data) if self.requires_grad else None
        )
        self._backward = lambda: None
        # Parents in the dynamic computation graph built during forward pass.
        self._prev = set(_children)
        self._op = _op

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    def item(self) -> float:
        if self.data.size != 1:
            raise ValueError("Only scalar tensor can be converted to Python float")
        return float(self.data.item())

    def numpy(self) -> np.ndarray:
        return self.data.copy()

    def zero_grad(self) -> None:
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)

    def detach(self) -> Tensor:
        return Tensor(self.data.copy(), requires_grad=False)

    @staticmethod
    def _coerce(other: Any) -> Tensor:
        if isinstance(other, Tensor):
            return other
        return Tensor(other, requires_grad=False)

    def __repr__(self) -> str:
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad})"

    def __getstate__(self) -> dict[str, Any]:
        return {
            "data": self.data,
            "requires_grad": self.requires_grad,
            "grad": self.grad,
            "_op": self._op,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.data = state["data"]
        self.requires_grad = state["requires_grad"]
        self.grad = state["grad"]
        self._op = state.get("_op", "")
        self._backward = lambda: None
        self._prev = set()

    def __add__(self, other: Any) -> Tensor:
        other_t = self._coerce(other)
        out = Tensor(
            self.data + other_t.data,
            requires_grad=self.requires_grad or other_t.requires_grad,
            _children=(self, other_t),
            _op="+",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            # d(self + other)/dself = 1 and d(self + other)/dother = 1.
            if self.requires_grad and self.grad is not None:
                self.grad += _unbroadcast(out.grad, self.shape)
            if other_t.requires_grad and other_t.grad is not None:
                other_t.grad += _unbroadcast(out.grad, other_t.shape)

        out._backward = _backward
        return out

    def __radd__(self, other: Any) -> Tensor:
        return self + other

    def __sub__(self, other: Any) -> Tensor:
        return self + (-self._coerce(other))

    def __rsub__(self, other: Any) -> Tensor:
        return self._coerce(other) - self

    def __neg__(self) -> Tensor:
        out = Tensor(
            -self.data,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="neg",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            # If y = -x, then dL/dx = dL/dy * (-1).
            if self.requires_grad and self.grad is not None:
                self.grad -= out.grad

        out._backward = _backward
        return out

    def __mul__(self, other: Any) -> Tensor:
        other_t = self._coerce(other)
        out = Tensor(
            self.data * other_t.data,
            requires_grad=self.requires_grad or other_t.requires_grad,
            _children=(self, other_t),
            _op="*",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            # Product rule with shape correction when NumPy broadcasting was used.
            if self.requires_grad and self.grad is not None:
                self.grad += _unbroadcast(out.grad * other_t.data, self.shape)
            if other_t.requires_grad and other_t.grad is not None:
                other_t.grad += _unbroadcast(out.grad * self.data, other_t.shape)

        out._backward = _backward
        return out

    def __rmul__(self, other: Any) -> Tensor:
        return self * other

    def __truediv__(self, other: Any) -> Tensor:
        other_t = self._coerce(other)
        out = Tensor(
            self.data / other_t.data,
            requires_grad=self.requires_grad or other_t.requires_grad,
            _children=(self, other_t),
            _op="/",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            # If y = x / b, then:
            # dL/dx = dL/dy * (1 / b)
            # dL/db = dL/dy * (-x / b^2)
            if self.requires_grad and self.grad is not None:
                self.grad += _unbroadcast(out.grad / other_t.data, self.shape)
            if other_t.requires_grad and other_t.grad is not None:
                other_t.grad += _unbroadcast(
                    -out.grad * self.data / (other_t.data**2),
                    other_t.shape,
                )

        out._backward = _backward
        return out

    def __rtruediv__(self, other: Any) -> Tensor:
        return self._coerce(other) / self

    def __pow__(self, exponent: float) -> Tensor:
        out = Tensor(
            self.data**exponent,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op=f"**{exponent}",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            # If y = x^a for constant a, then dL/dx = dL/dy * a * x^(a-1).
            if self.requires_grad and self.grad is not None:
                self.grad += out.grad * (exponent * self.data ** (exponent - 1.0))

        out._backward = _backward
        return out

    def __matmul__(self, other: Any) -> Tensor:
        other_t = self._coerce(other)
        out = Tensor(
            self.data @ other_t.data,
            requires_grad=self.requires_grad or other_t.requires_grad,
            _children=(self, other_t),
            _op="@",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            # For z = x @ w:
            # dL/dx = dL/dz @ w^T and dL/dw = x^T @ dL/dz.
            if self.requires_grad and self.grad is not None:
                self.grad += out.grad @ other_t.data.T
            if other_t.requires_grad and other_t.grad is not None:
                other_t.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def sum(
        self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> Tensor:
        out = Tensor(
            np.sum(self.data, axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="sum",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            if not self.requires_grad or self.grad is None:
                return

            grad = out.grad
            if axis is None:
                # If y = sum(x), each x_i contributes with local derivative 1.
                grad = np.ones_like(self.data) * grad
            else:
                axes = axis if isinstance(axis, tuple) else (axis,)
                expanded = grad
                if not keepdims:
                    # Reinsert removed axes before broadcasting back to input shape.
                    for ax in sorted(axes):
                        expanded = np.expand_dims(expanded, ax)
                grad = np.ones_like(self.data) * expanded
            self.grad += grad

        out._backward = _backward
        return out

    def mean(
        self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> Tensor:
        # mean(x) is implemented as sum(x) / N, so backward reuses both rules.
        if axis is None:
            denom = self.data.size
        elif isinstance(axis, tuple):
            denom = int(np.prod([self.data.shape[a] for a in axis]))
        else:
            denom = self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) / denom

    def log(self) -> Tensor:
        out = Tensor(
            np.log(self.data),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="log",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            # If y = log(x), then dL/dx = dL/dy * 1/x.
            if self.requires_grad and self.grad is not None:
                self.grad += out.grad / self.data

        out._backward = _backward
        return out

    def exp(self) -> Tensor:
        out = Tensor(
            np.exp(self.data),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="exp",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            # If y = exp(x), then dL/dx = dL/dy * exp(x).
            if self.requires_grad and self.grad is not None:
                self.grad += out.grad * out.data

        out._backward = _backward
        return out

    def tanh(self) -> Tensor:
        t = np.tanh(self.data)
        out = Tensor(
            t,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="tanh",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            # If y = tanh(x), then dL/dx = dL/dy * (1 - tanh(x)^2).
            if self.requires_grad and self.grad is not None:
                self.grad += out.grad * (1.0 - t**2)

        out._backward = _backward
        return out

    def relu(self) -> Tensor:
        out_data = np.maximum(0.0, self.data)
        out = Tensor(
            out_data,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="relu",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            # ReLU has local derivative 1 for x > 0 and 0 otherwise.
            if self.requires_grad and self.grad is not None:
                self.grad += out.grad * (self.data > 0.0)

        out._backward = _backward
        return out

    def sigmoid(self) -> Tensor:
        clipped = np.clip(self.data, -500.0, 500.0)
        s = 1.0 / (1.0 + np.exp(-clipped))
        out = Tensor(
            s,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="sigmoid",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            # If y = sigmoid(x), then dL/dx = dL/dy * y * (1 - y).
            if self.requires_grad and self.grad is not None:
                self.grad += out.grad * s * (1.0 - s)

        out._backward = _backward
        return out

    def softmax(self, axis: int = -1) -> Tensor:
        shifted = self.data - np.max(self.data, axis=axis, keepdims=True)
        exp_values = np.exp(shifted)
        s = exp_values / np.sum(exp_values, axis=axis, keepdims=True)
        out = Tensor(
            s,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="softmax",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            if not self.requires_grad or self.grad is None:
                return
            # Jacobian-vector product for softmax:
            # dL/dx = s * (g - sum(g * s)), with g = dL/ds along the softmax axis.
            self.grad += s * (out.grad - np.sum(out.grad * s, axis=axis, keepdims=True))

        out._backward = _backward
        return out

    def clamp(self, min_value: float, max_value: float) -> Tensor:
        out_data = np.clip(self.data, min_value, max_value)
        out = Tensor(
            out_data,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="clamp",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad and self.grad is not None:
                # Clamp is locally identity inside [min_value, max_value] and flat outside.
                mask = (self.data >= min_value) & (self.data <= max_value)
                self.grad += out.grad * mask

        out._backward = _backward
        return out

    def backward(self, grad: np.ndarray | None = None) -> None:
        if not self.requires_grad:
            return

        if grad is None:
            if self.data.size != 1:
                raise RuntimeError("grad must be specified for non-scalar tensor")
            # For a scalar output y, backward starts with dy/dy = 1.
            grad = np.ones_like(self.data)

        self.grad = self.grad + grad if self.grad is not None else grad

        topo: list[Tensor] = []
        visited: set[int] = set()

        def build(node: Tensor) -> None:
            node_id = id(node)
            if node_id in visited:
                return
            visited.add(node_id)
            for child in node._prev:
                build(child)
            topo.append(node)

        # Build a topological order so every node is visited after its children.
        build(self)

        # Reverse traversal applies the chain rule from output back to leaves.
        for node in reversed(topo):
            node._backward()
