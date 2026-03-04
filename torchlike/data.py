from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from .tensor import Tensor


class ArrayDataset:
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)

        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(-1, 1)
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)
        if x_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("x and y must have the same number of samples")

        self.x = x_arr
        self.y = y_arr

    def __len__(self) -> int:
        return self.x.shape[0]

    def batch(self, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.x[indices], self.y[indices]


class DataLoader:
    def __init__(
        self,
        dataset: ArrayDataset,
        *,
        batch_size: int = 32,
        shuffle: bool = True,
        seed: int | None = None,
        drop_last: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.rng = np.random.default_rng(seed)

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            self.rng.shuffle(indices)

        for start in range(0, len(indices), self.batch_size):
            batch_indices = indices[start : start + self.batch_size]
            if self.drop_last and batch_indices.shape[0] < self.batch_size:
                continue
            x_batch, y_batch = self.dataset.batch(batch_indices)
            yield Tensor(x_batch), Tensor(y_batch)

    def __len__(self) -> int:
        total = len(self.dataset)
        if self.drop_last:
            return total // self.batch_size
        return (total + self.batch_size - 1) // self.batch_size
