from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List


from curricmeta.tasks.base import SupervisedCurriculumTask


class Curriculum(ABC):
    @abstractmethod
    def num_stages(self) -> int: ...

    @abstractmethod
    def make_stages(self, task: SupervisedCurriculumTask) -> List[Any]:
        """
        Return a list of curriculum stages for the task.
        For supervised tasks: usually DataLoader objects.
        """
        ...
