from __future__ import annotations
from abc import ABC
from typing import Any, Dict, Iterable, List, Mapping, Optional


class Callback(ABC):
    """Base class for all callbacks.

    All hooks are optional: subclasses override what they need.
    `state` is a mutable experiment state dict controlled by the experiment.
    """

    def __init__(self, config: Optional[Mapping[str, Any]] = None):
        self.config: Dict[str, Any] = dict(config or {})

    # Experiment-level hooks
    def on_experiment_start(self, cfg: Any, state: Dict[str, Any]) -> None:
        pass

    def on_experiment_end(self, cfg: Any, state: Dict[str, Any]) -> None:
        pass

    # Outer loop hooks
    def on_outer_step_start(self, outer_step: int, cfg: Any, state: Dict[str, Any]) -> None:
        pass

    def on_outer_step_end(
        self,
        outer_step: int,
        metrics: Dict[str, Any],
        cfg: Any,
        state: Dict[str, Any],
    ) -> None:
        pass

    # Inner loop hooks
    def on_inner_loop_start(self, outer_step: int, cfg: Any, state: Dict[str, Any]) -> None:
        pass

    def on_inner_loop_end(
        self,
        outer_step: int,
        inner_results: Dict[str, Any],
        cfg: Any,
        state: Dict[str, Any],
    ) -> None:
        pass


class CallbackList(Callback):
    """Composite callback that forwards hooks to a list of callbacks."""

    def __init__(self, callbacks: Iterable[Callback]):
        super().__init__(config={})
        self.callbacks: List[Callback] = list(callbacks)

    def on_experiment_start(self, cfg: Any, state: Dict[str, Any]) -> None:
        for cb in self.callbacks:
            cb.on_experiment_start(cfg, state)

    def on_experiment_end(self, cfg: Any, state: Dict[str, Any]) -> None:
        for cb in self.callbacks:
            cb.on_experiment_end(cfg, state)

    def on_outer_step_start(self, outer_step: int, cfg: Any, state: Dict[str, Any]) -> None:
        for cb in self.callbacks:
            cb.on_outer_step_start(outer_step, cfg, state)

    def on_outer_step_end(
        self,
        outer_step: int,
        metrics: Dict[str, Any],
        cfg: Any,
        state: Dict[str, Any],
    ) -> None:
        for cb in self.callbacks:
            cb.on_outer_step_end(outer_step, metrics, cfg, state)

    def on_inner_loop_start(self, outer_step: int, cfg: Any, state: Dict[str, Any]) -> None:
        for cb in self.callbacks:
            cb.on_inner_loop_start(outer_step, cfg, state)

    def on_inner_loop_end(
        self,
        outer_step: int,
        inner_results: Dict[str, Any],
        cfg: Any,
        state: Dict[str, Any],
    ) -> None:
        for cb in self.callbacks:
            cb.on_inner_loop_end(outer_step, inner_results, cfg, state)
