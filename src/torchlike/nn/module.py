from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from torchlike.tensor import Tensor


class Module:
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def _collect_parameters(self, obj: Any, out: list[Tensor], seen: set[int]) -> None:
        if isinstance(obj, Tensor) and obj.requires_grad:
            obj_id = id(obj)
            if obj_id not in seen:
                seen.add(obj_id)
                out.append(obj)
            return

        if isinstance(obj, Module):
            for param in obj.parameters():
                param_id = id(param)
                if param_id not in seen:
                    seen.add(param_id)
                    out.append(param)
            return

        if isinstance(obj, (list, tuple)):
            for item in obj:
                self._collect_parameters(item, out, seen)
            return

        if isinstance(obj, dict):
            for item in obj.values():
                self._collect_parameters(item, out, seen)

    def parameters(self) -> list[Tensor]:
        out: list[Tensor] = []
        seen: set[int] = set()
        for value in self.__dict__.values():
            self._collect_parameters(value, out, seen)
        return out

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()

    def save(self, path: str | Path) -> None:
        path = Path(path)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path):
        path = Path(path)
        with path.open("rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is {type(obj)}, expected {cls}")
        return obj
