from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from omegaconf import DictConfig

from .registries import (
    EXPERIMENT_REGISTRY,
    CALLBACK_REGISTRY,
    get_registered,
)
from .callbacks.base import Callback, CallbackList


@dataclass
class ExperimentConfig:
    """Thin wrapper around Hydra config, if you want to pass it around later."""
    hydra_cfg: DictConfig


def build_callbacks(cfg: DictConfig) -> CallbackList:
    """
    Instantiate callbacks from cfg.callbacks.

    Expected cfg.callbacks:
      - name: registry key in CALLBACK_REGISTRY
        params: dict passed to callback constructor
    """
    callbacks_cfg = getattr(cfg, "callbacks", [])
    instances: List[Callback] = []

    for cb_cfg in callbacks_cfg:
        cb_name = cb_cfg.name
        cb_params = dict(getattr(cb_cfg, "params", {}))
        cb_cls = get_registered(CALLBACK_REGISTRY, cb_name)
        instances.append(cb_cls(cb_params))

    return CallbackList(instances)


def run_experiment(cfg: DictConfig, callbacks: CallbackList | None = None) -> Dict[str, Any]:
    """
    Top-level experiment runner, conceptually similar to a Trainer in rlcap.

    - Uses the EXPERIMENT_REGISTRY to resolve cfg.experiment.name.
    - Calls the corresponding experiment function with (cfg, callbacks).
    """
    exp_name = cfg.experiment.name
    exp_fn = get_registered(EXPERIMENT_REGISTRY, exp_name)

    if callbacks is None:
        callbacks = build_callbacks(cfg)

    results: Dict[str, Any] = exp_fn(cfg, callbacks)
    return results
