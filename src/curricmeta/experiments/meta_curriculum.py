from __future__ import annotations

from typing import Any, Dict

import torch
from omegaconf import DictConfig

from curricmeta.callbacks.base import CallbackList
from curricmeta.tasks.base import SupervisedStagedTask
from curricmeta.utils.registry import register, build


@register("experiment", "meta_curriculum")
class MetaCurriculumExperiment:
    """
    Generic meta-learning experiment for staged supervised tasks with curricula.

    Config controls:
      - task.name
      - curriculum.name
      - model.name
      - meta.inner_loop.name
      - meta.outer_loop.name
    """

    def __init__(self, cfg: DictConfig, callbacks: CallbackList):
        self.cfg = cfg
        self.callbacks = callbacks

    def run(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {}
        self.callbacks.on_experiment_start(self.cfg, state)

        device_str = str(getattr(self.cfg.experiment, "device", "cpu"))
        device = torch.device(device_str)

        # Builders used by the outer loop
        def build_task() -> SupervisedStagedTask:
            task_cfg = dict(self.cfg.task)
            name = task_cfg.pop("name")
            return build("task", name, config=task_cfg)

        def build_curriculum():
            cur_cfg = dict(self.cfg.curriculum)
            name = cur_cfg.pop("name")
            return build("curriculum", name, config=cur_cfg)

        def build_model(task: SupervisedStagedTask) -> torch.nn.Module:
            model_cfg = dict(self.cfg.model)
            name = model_cfg.pop("name")
            model_cfg.setdefault("in_dim", task.input_dim())
            model_cfg.setdefault("n_classes", task.num_classes())
            model_cfg.setdefault("out_dim", task.num_classes())
            return build("model", name, config=model_cfg)

        # Inner loop
        inner_name = self.cfg.meta.inner_loop.name
        inner_cfg = dict(self.cfg.meta.inner_loop)
        inner_loop = build("inner_loop", inner_name, config=inner_cfg)

        # Outer loop
        outer_name = self.cfg.meta.outer_loop.name
        outer_cfg = dict(self.cfg.meta.outer_loop)
        outer_loop = build(
            "outer_loop",
            outer_name,
            config=outer_cfg,
            inner_loop=inner_loop,
            build_task=build_task,
            build_curriculum=build_curriculum,
            build_model=build_model,
            device=device,
        )

        results = outer_loop.run()

        self.callbacks.on_experiment_end(self.cfg, state)
        return results
