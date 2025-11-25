from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict
from torch.utils.data import DataLoader


class Task(ABC):
    """
    Minimal base Task interface.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def setup(self) -> None:
        """Allocate resources / build internal state."""
        ...

    @abstractmethod
    def teardown(self) -> None:
        """Clean up resources if needed."""
        ...


class SupervisedTask(Task):
    """
    Base class for supervised tasks. Provides dimensionality info used to
    build models, but no curriculum or staging logic.
    """

    @abstractmethod
    def input_dim(self) -> int:
        ...

    @abstractmethod
    def num_classes(self) -> int:
        ...


class SupervisedStagedTask(SupervisedTask):
    """
    Supervised task that exposes a finite set of 'stages' (e.g. difficulty
    levels). The curriculum will *schedule* these stages, but *not* handle
    sampling itself.
    """

    @abstractmethod
    def num_stages(self) -> int:
        """Number of curriculum stages the task exposes."""
        ...

    @abstractmethod
    def get_train_loader(self, stage_id: int) -> DataLoader:
        """Return the train loader for a given stage index."""
        ...

    @abstractmethod
    def get_eval_loader(self, stage_id: int | None = None) -> DataLoader:
        """
        Return the eval loader. If stage_id is None, the task can choose a
        default (e.g. 'hardest' or 'final' stage).
        """
        ...
