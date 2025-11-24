# src/curricmeta/experiments/dummy_experiment.py

from typing import Dict, Any
from omegaconf import DictConfig

from curricmeta.callbacks.base import CallbackList
from curricmeta.utils.registry import register


@register("experiment", "dummy_experiment")
class DummyExperiment:
    def __init__(self, cfg: DictConfig, callbacks: CallbackList):
        self.cfg = cfg
        self.callbacks = callbacks

    def run(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {}
        self.callbacks.on_experiment_start(self.cfg, state)

        n_outer = self.cfg.experiment.n_outer_steps
        for outer_step in range(n_outer):
            self.callbacks.on_outer_step_start(outer_step, self.cfg, state)
            self.callbacks.on_inner_loop_start(outer_step, self.cfg, state)

            inner_results = {"dummy": True, "outer_step": outer_step}
            self.callbacks.on_inner_loop_end(outer_step, inner_results, self.cfg, state)

            metrics = {"meta_loss": 0.0, "outer_step": outer_step}
            self.callbacks.on_outer_step_end(outer_step, metrics, self.cfg, state)

        self.callbacks.on_experiment_end(self.cfg, state)
        return {"final_metrics": {"meta_loss": 0.0}}
