from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from curricmeta.curriculum.base import Curriculum
from curricmeta.tasks.gaussian_2d import Gaussian2DTask
from curricmeta.utils.registry import register


@register("curriculum", "gaussian_easy_to_hard")
class GaussianEasyToHard(Curriculum):
    """
    Easy->hard curriculum for Gaussian2DTask.

    Stages differ by standard deviation (std) of the Gaussians.
    """

    def __init__(self, config: Dict[str, Any]):
        self.stage_stds: List[float] = list(
            config.get("stage_stds", [0.05, 0.15, 0.3, 0.5])
        )
        self.n_samples_per_stage: int = int(config.get("n_samples_per_stage", 2000))
        self.batch_size: int = int(config.get("batch_size", 128))

        self._train_loaders: List[DataLoader] = []
        self._eval_loader: DataLoader | None = None

    def setup(self, task: Gaussian2DTask) -> None:
        rng = np.random.default_rng(task.seed)
        self._train_loaders.clear()

        # One stage per std
        for std in self.stage_stds:
            X, y = task.sample_stage(
                rng, std=std, n_samples_per_class=self.n_samples_per_stage
            )
            ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
            loader = DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=True,
            )
            self._train_loaders.append(loader)

        # Eval from hardest distribution (last std)
        X_eval, y_eval = task.sample_stage(
            rng,
            std=self.stage_stds[-1],
            n_samples_per_class=self.n_samples_per_stage,
        )
        eval_ds = TensorDataset(torch.from_numpy(X_eval), torch.from_numpy(y_eval))
        self._eval_loader = DataLoader(
            eval_ds,
            batch_size=self.batch_size,
            shuffle=False,
        )

    def teardown(self) -> None:
        self._train_loaders.clear()
        self._eval_loader = None

    def num_stages(self) -> int:
        return len(self._train_loaders)

    def train_loaders(self) -> List[DataLoader]:
        return self._train_loaders

    def eval_loader(self) -> DataLoader:
        assert self._eval_loader is not None, "Call setup(task) before eval_loader()"
        return self._eval_loader
