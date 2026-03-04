from __future__ import annotations

from typing import Iterable

from torchlike.tensor import Tensor

from .module import Module
import matplotlib.pyplot as plt


class Sequential(Module):
    def __init__(self, layers: Iterable[Module]) -> None:
        self.layers = list(layers)

    def forward(self, x: Tensor) -> Tensor:
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, idx: int) -> Module:
        return self.layers[idx]

    def plot_weight_distribution(self, layer_indices: list[int]) -> None:
        self._plot_distribution(layer_indices, use_grad=False)

    def plot_gradient_distribution(self, layer_indices: list[int]) -> None:
        self._plot_distribution(layer_indices, use_grad=True)

    def _plot_distribution(self, layer_indices: list[int], *, use_grad: bool) -> None:

        linear_layers = [layer for layer in self.layers if hasattr(layer, "weight")]
        if not layer_indices:
            raise ValueError("layer_indices cannot be empty")
        for idx in layer_indices:
            if idx < 0 or idx >= len(linear_layers):
                raise IndexError(f"Layer index out of range: {idx}")

        fig, axes = plt.subplots(
            1, len(layer_indices), figsize=(5 * len(layer_indices), 4)
        )
        if len(layer_indices) == 1:
            axes = [axes]

        title_prefix = "Gradient" if use_grad else "Weight"
        for ax, idx in zip(axes, layer_indices):
            layer = linear_layers[idx]
            weight_values = layer.weight.grad if use_grad else layer.weight.data
            values = [weight_values.reshape(-1)]
            if getattr(layer, "bias", None) is not None:
                bias_values = layer.bias.grad if use_grad else layer.bias.data
                values.append(bias_values.reshape(-1))

            import numpy as np

            flat = np.concatenate(values)
            ax.hist(flat, bins=30, edgecolor="black", alpha=0.85)
            ax.set_title(f"{title_prefix} Dist Layer {idx}")
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")

        plt.tight_layout()
        plt.show()
