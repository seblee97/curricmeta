from __future__ import annotations

from typing import List, Any, Dict
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from curricmeta.curriculum.base import Curriculum
from curricmeta.utils.registry import register
from curricmeta.tasks.gaussian_2d import Gaussian2DTask


@register("curriculum", "gaussian_easy_to_hard")
class GaussianEasyToHard(Curriculum):

    def __init__(self, config: Dict[str, Any]):
        self.stds: List[float] = list(config.get("stage_stds", [0.05, 0.15, 0.3, 0.5]))
        self.n_samples_per_stage = int(config.get("n_samples_per_stage", 2000))
        self.batch_size = int(config.get("batch_size", 128))

    def num_stages(self) -> int:
        return len(self.stds)

    def make_stages(self, task: Gaussian2DTask) -> List[DataLoader]:
        loaders = []
        rng = np.random.default_rng(task.seed)

        for std in self.stds:
            X, y = task.make_samples(rng, std, self.n_samples_per_stage)
            ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
            loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)
            loaders.append(loader)

        return loaders
