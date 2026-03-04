from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from typing import Any, TypeAlias

from torchlike.tensor import Tensor

StateDict: TypeAlias = dict[str, Any]


class Optimizer:
    def __init__(self, params: Iterable[Tensor], defaults: dict[str, Any]) -> None:
        if isinstance(params, Tensor):
            raise TypeError(
                "params argument given to the optimizer should be an iterable of Tensors"
            )

        param_list = list(params)
        if len(param_list) == 0:
            raise ValueError("optimizer got an empty parameter list")
        for p in param_list:
            if not isinstance(p, Tensor):
                raise TypeError(f"optimizer can only optimize Tensors, got {type(p)}")

        self.defaults = defaults.copy()
        self.state: dict[Tensor, dict[str, Any]] = {}
        self.param_groups: list[dict[str, Any]] = []
        self.add_param_group({"params": param_list})

    @property
    def params(self) -> list[Tensor]:
        all_params: list[Tensor] = []
        for group in self.param_groups:
            all_params.extend(group["params"])
        return all_params

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + " ("
        for i, group in enumerate(self.param_groups):
            format_string += f"\nParameter Group {i}\n"
            for key in sorted(group.keys()):
                if key != "params":
                    format_string += f"    {key}: {group[key]}\n"
        format_string += ")"
        return format_string

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        if not isinstance(param_group, dict):
            raise TypeError(f"param_group must be a dict, got {type(param_group)}")
        if "params" not in param_group:
            raise ValueError("param_group must contain 'params'")

        params = param_group["params"]
        if isinstance(params, Tensor):
            params = [params]
        else:
            params = list(params)
        if len(params) == 0:
            raise ValueError("optimizer got an empty parameter group")

        existing = {id(p) for group in self.param_groups for p in group["params"]}
        for p in params:
            if not isinstance(p, Tensor):
                raise TypeError(f"optimizer can only optimize Tensors, got {type(p)}")
            if id(p) in existing:
                raise ValueError(
                    "some parameters appear in more than one parameter group"
                )

        group = {k: deepcopy(v) for k, v in self.defaults.items()}
        for k, v in param_group.items():
            if k != "params":
                group[k] = v
        group["params"] = params
        self.param_groups.append(group)

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()

    def state_dict(self) -> StateDict:
        param_to_idx = {id(p): i for i, p in enumerate(self.params)}
        packed_groups: list[dict[str, Any]] = []
        for group in self.param_groups:
            packed = {k: deepcopy(v) for k, v in group.items() if k != "params"}
            packed["params"] = [param_to_idx[id(p)] for p in group["params"]]
            packed_groups.append(packed)

        packed_state: dict[int, dict[str, Any]] = {}
        for p, s in self.state.items():
            idx = param_to_idx.get(id(p))
            if idx is not None:
                packed_state[idx] = deepcopy(s)

        return {
            "defaults": deepcopy(self.defaults),
            "state": packed_state,
            "param_groups": packed_groups,
        }

    def load_state_dict(self, state_dict: StateDict) -> None:
        if "param_groups" not in state_dict or "state" not in state_dict:
            raise ValueError("invalid state_dict: missing keys")

        loaded_groups = state_dict["param_groups"]
        if len(loaded_groups) != len(self.param_groups):
            raise ValueError(
                "loaded state dict has a different number of parameter groups"
            )

        for g_loaded, g_current in zip(loaded_groups, self.param_groups, strict=True):
            if len(g_loaded["params"]) != len(g_current["params"]):
                raise ValueError(
                    "loaded state dict contains a parameter group with a different size"
                )

        for g_loaded, g_current in zip(loaded_groups, self.param_groups, strict=True):
            for key, value in g_loaded.items():
                if key != "params":
                    g_current[key] = deepcopy(value)

        self.defaults = deepcopy(state_dict.get("defaults", self.defaults))

        idx_to_param = {i: p for i, p in enumerate(self.params)}
        restored_state: dict[Tensor, dict[str, Any]] = {}
        for idx, value in state_dict["state"].items():
            if int(idx) in idx_to_param:
                restored_state[idx_to_param[int(idx)]] = deepcopy(value)
        self.state = restored_state

    def step(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError
