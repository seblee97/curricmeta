from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from curricmeta.tasks.base import SupervisedStagedTask
from curricmeta.utils.registry import register


@register("task", "gaussian_2d")
class Gaussian2DTask(SupervisedStagedTask):
    """
    2D Gaussian classification task with a built-in notion of staged
    difficulty (via std values). The *task* handles sampling; the
    *curriculum* only decides how to order the stage indices.

    Stage i corresponds to std = stage_stds[i].
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.n_classes: int = int(config.get("n_classes", 3))
        self.dim: int = int(config.get("dim", 2))
        assert self.dim == 2, "Gaussian2DTask currently assumes dim=2"

        self.mean_scale: float = float(config.get("mean_scale", 2.0))
        self.stage_stds: List[float] = list(
            config.get("stage_stds", [0.05, 0.15, 0.3, 0.5])
        )
        self.n_samples_per_stage: int = int(config.get("n_samples_per_stage", 2000))
        self.batch_size: int = int(config.get("batch_size", 128))
        self.seed: int = int(config.get("seed", 1234))

        self._train_loaders: List[DataLoader] = []
        self._eval_loader: DataLoader | None = None

        # Precompute class means on a circle
        angles = np.linspace(0.0, 2 * np.pi, self.n_classes, endpoint=False)
        self.means = np.stack(
            [self.mean_scale * np.cos(angles), self.mean_scale * np.sin(angles)],
            axis=1,
        )  # (K, 2)

    # ---------- internal sampling helper ----------

    def _sample_stage(
        self,
        rng: np.random.Generator,
        std: float,
        n_samples_per_class: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        xs = []
        ys = []
        cov = (std ** 2) * np.eye(2)

        for cls in range(self.n_classes):
            mean = self.means[cls]
            samples = rng.multivariate_normal(mean, cov, size=n_samples_per_class)
            xs.append(samples)
            ys.append(np.full((n_samples_per_class,), cls, dtype=np.int64))

        X = np.concatenate(xs, axis=0).astype(np.float32)
        y = np.concatenate(ys, axis=0).astype(np.int64)
        return torch.from_numpy(X), torch.from_numpy(y)

    # ---------- Task base methods ----------

    def setup(self) -> None:
        rng = np.random.default_rng(self.seed)
        self._train_loaders.clear()

        for std in self.stage_stds:
            X, y = self._sample_stage(
                rng, std=std, n_samples_per_class=self.n_samples_per_stage
            )
            ds = TensorDataset(X, y)
            loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)
            self._train_loaders.append(loader)

        # Eval from hardest distribution (last stage)
        X_eval, y_eval = self._sample_stage(
            rng,
            std=self.stage_stds[-1],
            n_samples_per_class=self.n_samples_per_stage,
        )
        eval_ds = TensorDataset(X_eval, y_eval)
        self._eval_loader = DataLoader(
            eval_ds, batch_size=self.batch_size, shuffle=False
        )

    def teardown(self) -> None:
        self._train_loaders.clear()
        self._eval_loader = None

    # ---------- SupervisedTask interface ----------

    def input_dim(self) -> int:
        return 2

    def num_classes(self) -> int:
        return self.n_classes

    # ---------- SupervisedStagedTask interface ----------

    def num_stages(self) -> int:
        return len(self.stage_stds)

    def get_train_loader(self, stage_id: int) -> DataLoader:
        return self._train_loaders[stage_id]

    def get_eval_loader(self, stage_id: int | None = None) -> DataLoader:
        # For now, ignore stage_id and always evaluate on 'hardest' distribution.
        assert self._eval_loader is not None, "Call setup() before get_eval_loader()"
        return self._eval_loader
