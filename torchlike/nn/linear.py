from __future__ import annotations

import numpy as np

import torchlike.functional as F
from torchlike.tensor import Tensor

from .module import Module


def _initialize(
    shape: tuple[int, ...],
    *,
    method: str,
    params: dict[str, float | str],
    rng: np.random.Generator,
) -> np.ndarray:
    method = method.lower().strip()
    if method == "zero":
        return np.zeros(shape, dtype=np.float64)

    if method == "uniform":
        lower = float(params.get("lower_bound", -0.1))
        upper = float(params.get("upper_bound", 0.1))
        return rng.uniform(lower, upper, size=shape).astype(np.float64)

    if method == "normal":
        mean = float(params.get("mean", 0.0))
        variance = float(params.get("variance", 0.01))
        if variance < 0:
            raise ValueError("variance must be non-negative")
        return rng.normal(mean, np.sqrt(variance), size=shape).astype(np.float64)

    fan_in = shape[0]
    fan_out = shape[1] if len(shape) > 1 else shape[0]
    if fan_in <= 0:
        raise ValueError("fan_in must be positive")

    if method in ("xavier", "xavier_uniform", "xavier_normal"):
        gain = float(params.get("gain", 1.0))
        if method == "xavier_normal":
            std = gain * np.sqrt(2.0 / (fan_in + fan_out))
            return rng.normal(0.0, std, size=shape).astype(np.float64)
        limit = gain * np.sqrt(6.0 / (fan_in + fan_out))
        return rng.uniform(-limit, limit, size=shape).astype(np.float64)

    if method in ("he", "he_uniform", "he_normal"):
        if method == "he_uniform":
            limit = np.sqrt(6.0 / fan_in)
            return rng.uniform(-limit, limit, size=shape).astype(np.float64)
        std = np.sqrt(2.0 / fan_in)
        return rng.normal(0.0, std, size=shape).astype(np.float64)

    raise ValueError(f"Unsupported initialization method: {method}")


class Linear(Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        init: str = "uniform",
        init_params: dict[str, float | str] | None = None,
        seed: int | None = None,
    ) -> None:
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.use_bias = bool(bias)
        self.init = init
        self.init_params: dict[str, float | str] = init_params or {}
        self.seed = seed

        rng = np.random.default_rng(seed)
        self.weight = Tensor(
            _initialize(
                (self.in_features, self.out_features),
                method=init,
                params=self.init_params,
                rng=rng,
            ),
            requires_grad=True,
        )
        self.bias = (
            Tensor(
                _initialize(
                    (1, self.out_features),
                    method=(
                        "zero"
                        if init.lower().strip()
                        in (
                            "xavier",
                            "xavier_uniform",
                            "xavier_normal",
                            "he",
                            "he_uniform",
                            "he_normal",
                        )
                        else init
                    ),
                    params={}
                    if init.lower().strip()
                    in (
                        "xavier",
                        "xavier_uniform",
                        "xavier_normal",
                        "he",
                        "he_uniform",
                        "he_normal",
                    )
                    else self.init_params,
                    rng=rng,
                ),
                requires_grad=True,
            )
            if self.use_bias
            else None
        )

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight, self.bias)
