from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List
from torch.utils.data import DataLoader


class Task(ABC):
    """Abstract task interface.

    Concrete tasks will:
    - provide dataset(s) / data generators for inner loop
    - define evaluation protocol for outer loop
    - optionally expose curriculum stages
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def setup(self) -> None:
        """Allocate resources, build datasets, etc."""
        ...

    @abstractmethod
    def teardown(self) -> None:
        """Clean up resources if needed."""
        ...


class SupervisedCurriculumTask(Task):
    """
    Generic interface for supervised tasks with a curriculum.

    Inner loops for supervised learning only depend on *this*,
    not on specific task implementations.
    """

    @abstractmethod
    def num_stages(self) -> int:
        """Number of curriculum stages."""
        ...

    @abstractmethod
    def train_loaders(self) -> List[DataLoader]:
        """One train loader per stage (0 .. num_stages-1)."""
        ...

    @abstractmethod
    def eval_loader(self) -> DataLoader:
        """Evaluation loader (often from the hardest or final distribution)."""
        ...

    @abstractmethod
    def input_dim(self) -> int:
        ...

    @abstractmethod
    def num_classes(self) -> int:
        ...
