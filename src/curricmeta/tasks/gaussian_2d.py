from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from curricmeta.tasks.base import SupervisedCurriculumTask
from curricmeta.utils.registry import register


@dataclass
class GaussianStageConfig:
    mean_scale: float  # distance of class means from origin
    std: float         # standard deviation of each class gaussian
    n_samples: int     # samples per class for this stage


@register("task", "gaussian_2d")
class Gaussian2DTask(SupervisedCurriculumTask):
    """
    Simple 2D Gaussian classification with an easy->hard curriculum.

    - Input: 2D points.
    - Classes: K gaussians located on a circle.
    - Curriculum hardness is controlled via per-stage std.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.n_classes: int = int(config.get("n_classes", 3))
        self.dim: int = int(config.get("dim", 2))
        assert self.dim == 2, "Gaussian2DTask currently assumes dim=2"

        self.n_samples_per_stage: int = int(config.get("n_samples_per_stage", 2000))
        self.batch_size: int = int(config.get("batch_size", 128))

        # Easy->hard std schedule; can be overridden in config
        default_stds = [0.05, 0.15, 0.3, 0.5]
        self.stage_stds: List[float] = list(config.get("stage_stds", default_stds))
        self.mean_scale: float = float(config.get("mean_scale", 2.0))

        self._train_loaders: List[DataLoader] = []
        self._eval_loader: DataLoader | None = None

    def _make_stage(
        self,
        rng: np.random.Generator,
        std: float,
        n_samples: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Place class means uniformly on a circle of radius mean_scale
        angles = np.linspace(0.0, 2 * np.pi, self.n_classes, endpoint=False)
        means = np.stack(
            [self.mean_scale * np.cos(angles), self.mean_scale * np.sin(angles)],
            axis=1,
        )  # (K, 2)

        xs = []
        ys = []

        for cls in range(self.n_classes):
            mean = means[cls]
            cov = (std ** 2) * np.eye(2)
            samples = rng.multivariate_normal(mean, cov, size=n_samples)
            xs.append(samples)
            ys.append(np.full((n_samples,), cls, dtype=np.int64))

        x = np.concatenate(xs, axis=0).astype(np.float32)
        y = np.concatenate(ys, axis=0).astype(np.int64)

        x_t = torch.from_numpy(x)
        y_t = torch.from_numpy(y)
        return x_t, y_t

    def setup(self) -> None:
        rng = np.random.default_rng(int(self.config.get("seed", 1234)))

        self._train_loaders.clear()
        for std in self.stage_stds:
            x, y = self._make_stage(rng, std, self.n_samples_per_stage)
            ds = TensorDataset(x, y)
            loader = DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=True,
            )
            self._train_loaders.append(loader)

        # Test set from hardest distribution
        x_test, y_test = self._make_stage(
            rng,
            std=self.stage_stds[-1],
            n_samples=self.n_samples_per_stage,
        )
        test_ds = TensorDataset(x_test, y_test)
        self._eval_loader = DataLoader(
            test_ds,
            batch_size=self.batch_size,
            shuffle=False,
        )

    def teardown(self) -> None:
        self._train_loaders.clear()
        self._eval_loader = None

    # Convenience accessors used by the experiment
    @property
    def train_loaders(self) -> List[DataLoader]:
        return self._train_loaders
    
    def num_stages(self) -> int:
        return len(self._train_loaders)

    @property
    def eval_loader(self) -> DataLoader:
        assert self._eval_loader is not None
        return self._eval_loader

    def input_dim(self) -> int:
        return 2  # for this task

    def num_classes(self) -> int:
        return int(self.config.get("n_classes", 3))
