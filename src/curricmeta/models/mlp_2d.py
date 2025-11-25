from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn

from curricmeta.models.base import Model
from curricmeta.utils.registry import register


def _make_mlp(
    in_dim: int,
    hidden_dims: List[int],
    out_dim: int,
    activation: str = "relu",
) -> nn.Module:
    acts = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "gelu": nn.GELU,
    }
    act_cls = acts.get(activation.lower(), nn.ReLU)

    layers: List[nn.Module] = []
    last_dim = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(last_dim, h))
        layers.append(act_cls())
        last_dim = h
    layers.append(nn.Linear(last_dim, out_dim))
    return nn.Sequential(*layers)


@register("model", "mlp_2d")
class MLP2D(nn.Module):
    """
    Simple MLP for 2D inputs and multi-class classification.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.in_dim: int = int(config.get("in_dim", 2))
        self.out_dim: int = int(config.get("out_dim", config.get("n_classes", 3)))
        self.hidden_dims: List[int] = list(config.get("hidden_dims", [64, 64]))
        self.activation: str = str(config.get("activation", "relu"))

        self.net = _make_mlp(
            in_dim=self.in_dim,
            hidden_dims=self.hidden_dims,
            out_dim=self.out_dim,
            activation=self.activation,
        )

    def initialize(self, rng: Any | None = None) -> None:
        """
        For now we just use PyTorch default initialization.
        This method exists to satisfy the Model interface.
        """
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
