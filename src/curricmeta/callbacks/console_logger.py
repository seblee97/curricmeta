from __future__ import annotations
from typing import Any, Dict, Mapping, Optional
import logging

from curricmeta.callbacks.base import Callback
from curricmeta.utils.registry import register

log = logging.getLogger(__name__)


@register("callbacks", "console_logger")
class ConsoleLoggerCallback(Callback):
    """Simple console logger to demonstrate callback wiring."""

    def __init__(self, config: Optional[Mapping[str, Any]] = None):
        super().__init__(config=config)

    def on_experiment_start(self, cfg: Any, state: Dict[str, Any]) -> None:
        log.info("Experiment started with config: %s", cfg.experiment.name)

    def on_experiment_end(self, cfg: Any, state: Dict[str, Any]) -> None:
        log.info("Experiment ended. Final state keys: %s", list(state.keys()))

    def on_outer_step_start(self, outer_step: int, cfg: Any, state: Dict[str, Any]) -> None:
        log.info("Outer step %d started.", outer_step)

    def on_outer_step_end(
        self,
        outer_step: int,
        metrics: Dict[str, Any],
        cfg: Any,
        state: Dict[str, Any],
    ) -> None:
        log.info("Outer step %d ended. Metrics: %s", outer_step, metrics)

    def on_inner_loop_start(self, outer_step: int, cfg: Any, state: Dict[str, Any]) -> None:
        log.info("Inner loop for outer step %d started.", outer_step)

    def on_inner_loop_end(
        self,
        outer_step: int,
        inner_results: Dict[str, Any],
        cfg: Any,
        state: Dict[str, Any],
    ) -> None:
        log.info("Inner loop for outer step %d ended. Inner results keys: %s",
                 outer_step, list(inner_results.keys()))
