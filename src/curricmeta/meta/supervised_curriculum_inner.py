from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn
from torch.optim import SGD

from curricmeta.meta.inner_loop_base import InnerLoop
from curricmeta.tasks.base import SupervisedStagedTask
from curricmeta.curriculum.base import Curriculum
from curricmeta.utils.registry import register


@register("inner_loop", "supervised_curriculum")
class SupervisedCurriculumInnerLoop(InnerLoop):
    """
    Generic inner loop for staged supervised tasks with a curriculum
    scheduler.

    Depends only on:
      - SupervisedStagedTask
      - Curriculum
      - model (nn.Module)
      - meta_params (e.g. per-stage learning rates)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # epochs per *stage index*
        self.n_epochs_per_stage: List[int] = list(
            config.get("n_epochs_per_stage", [5, 5, 5, 5])
        )

    def run(
        self,
        task: SupervisedStagedTask,
        curriculum: Curriculum,
        model: nn.Module,
        meta_params: Dict[str, Any],
        device: torch.device,
    ) -> Dict[str, Any]:
        """
        meta_params:
          - "per_stage_lr": list[float], length == task.num_stages()
        """
        task.setup()
        try:
            num_stages = task.num_stages()
            schedule = curriculum.build_schedule(num_stages)

            per_stage_lr: List[float] = list(meta_params["per_stage_lr"])
            assert len(per_stage_lr) == num_stages, (
                f"per_stage_lr length {len(per_stage_lr)} != num_stages {num_stages}"
            )

            # If n_epochs_per_stage shorter, pad; if longer, truncate.
            if len(self.n_epochs_per_stage) < num_stages:
                self.n_epochs_per_stage = (
                    self.n_epochs_per_stage
                    + [self.n_epochs_per_stage[-1]] * (num_stages - len(self.n_epochs_per_stage))
                )
            elif len(self.n_epochs_per_stage) > num_stages:
                self.n_epochs_per_stage = self.n_epochs_per_stage[:num_stages]

            model.to(device)
            model.train()
            criterion = nn.CrossEntropyLoss()

            stage_train_losses: List[float] = []

            for stage_id in schedule:
                lr = float(per_stage_lr[stage_id])
                n_epochs = int(self.n_epochs_per_stage[stage_id])
                loader = task.get_train_loader(stage_id)

                optimizer = SGD(model.parameters(), lr=lr, weight_decay=0.0)

                for _ in range(n_epochs):
                    running_loss = 0.0
                    n_batches = 0

                    for x, y in loader:
                        x = x.to(device)
                        y = y.to(device)

                        optimizer.zero_grad(set_to_none=True)
                        logits = model(x)
                        loss = criterion(logits, y)
                        loss.backward()
                        optimizer.step()

                        running_loss += loss.item()
                        n_batches += 1

                    stage_loss = running_loss / max(n_batches, 1)
                    stage_train_losses.append(stage_loss)

            # Evaluation (task decides what distribution to use)
            model.eval()
            eval_loader = task.get_eval_loader()
            correct = 0
            total = 0
            with torch.no_grad():
                for x, y in eval_loader:
                    x = x.to(device)
                    y = y.to(device)
                    logits = model(x)
                    preds = torch.argmax(logits, dim=-1)
                    correct += (preds == y).sum().item()
                    total += y.numel()

            test_acc = correct / max(total, 1)

            return {
                "stage_train_losses": stage_train_losses,
                "test_acc": test_acc,
            }
        finally:
            task.teardown()
