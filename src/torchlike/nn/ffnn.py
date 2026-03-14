from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

import numpy as np

from torchlike.tensor import Tensor

from .activations import GELU, SELU, Identity, ReLU, Sigmoid, Softmax, Tanh
from .linear import Linear
from .module import Module
from .sequential import Sequential

ActivationSpec: TypeAlias = str | Module | type[Module] | None


def _clone_activation(spec: ActivationSpec) -> Module | None:
    if spec is None:
        return None

    if isinstance(spec, Module):
        return spec

    if isinstance(spec, type) and issubclass(spec, Module):
        return spec()

    if not isinstance(spec, str):
        raise TypeError(f"Unsupported activation spec: {type(spec)}")

    name = spec.lower().strip()
    activation_map: dict[str, type[Module]] = {
        "linear": Identity,
        "identity": Identity,
        "relu": ReLU,
        "sigmoid": Sigmoid,
        "tanh": Tanh,
        "softmax": Softmax,
        "gelu": GELU,
        "selu": SELU,
    }
    if name not in activation_map:
        raise ValueError(f"Unsupported activation: {spec}")
    return activation_map[name]()


class FFNN(Module):
    def __init__(
        self,
        layer_sizes: Sequence[int],
        *,
        activations: Sequence[ActivationSpec] | None = None,
        bias: bool = True,
        init: str = "uniform",
        init_params: dict[str, float | str] | None = None,
        seed: int | None = None,
    ) -> None:
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must contain at least input and output size")

        self.layer_sizes = [int(size) for size in layer_sizes]
        if any(size <= 0 for size in self.layer_sizes):
            raise ValueError("layer_sizes must contain only positive integers")

        if activations is None:
            self.activations = [None] * (len(self.layer_sizes) - 1)
        else:
            self.activations = list(activations)
        if len(self.activations) != len(self.layer_sizes) - 1:
            raise ValueError("activations must have one entry for each non-input layer")

        self.bias = bool(bias)
        self.init = init
        self.init_params = dict(init_params or {})
        self.seed = seed

        layer_seed_rng = np.random.default_rng(seed) if seed is not None else None
        layers: list[Module] = []

        for idx, (in_features, out_features, activation) in enumerate(
            zip(
                self.layer_sizes[:-1],
                self.layer_sizes[1:],
                self.activations,
                strict=True,
            )
        ):
            layer_seed = None
            if layer_seed_rng is not None:
                layer_seed = int(layer_seed_rng.integers(0, np.iinfo(np.int32).max))

            layers.append(
                Linear(
                    in_features,
                    out_features,
                    bias=self.bias,
                    init=self.init,
                    init_params=self.init_params,
                    seed=layer_seed,
                )
            )

            activation_module = _clone_activation(activation)
            if activation_module is not None:
                layers.append(activation_module)

        self.network = Sequential(layers)

    @property
    def layers(self) -> list[Module]:
        return self.network.layers

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)
