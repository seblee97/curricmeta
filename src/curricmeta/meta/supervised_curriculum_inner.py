from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn
from torch.optim import SGD

from curricmeta.meta.inner_loop_base import InnerLoop
from curricmeta.tasks.base import SupervisedCurriculumTask
from curricmeta.utils.registry import register


@register("inner_loop", "supervised_curriculum")
class SupervisedCurriculumInnerLoop(InnerLoop):
    """
    Generic inner loop for supervised curriculum tasks.

    Depends only on:
      - SupervisedCurriculumTask
      - torch.nn.Module-like model
      - meta_params (e.g. per-stage lr)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.n_epochs_per_stage: List[int] = list(
            config.get("n_epochs_per_stage", [5, 5, 5, 5])
        )

    def run(
        self,
        task: SupervisedCurriculumTask,
        model: nn.Module,
        meta_params: Dict[str, Any],
        device: torch.device,
    ) -> Dict[str, Any]:
        """
        meta_params:
          - "per_stage_lr": list[float] of len task.num_stages()
        """
        model.to(device)
        model.train()

        train_loaders = task.train_loaders
        n_stages = task.num_stages

        per_stage_lr: List[float] = list(meta_params["per_stage_lr"])
        assert len(per_stage_lr) == n_stages
        assert len(self.n_epochs_per_stage) == n_stages

        criterion = nn.CrossEntropyLoss()
        stage_train_losses: List[float] = []

        for stage_idx in range(n_stages):
            lr = float(per_stage_lr[stage_idx])
            n_epochs = int(self.n_epochs_per_stage[stage_idx])
            loader = train_loaders[stage_idx]

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

        # Evaluation
        model.eval()
        eval_loader = task.eval_loader()
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
