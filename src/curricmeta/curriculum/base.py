from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable


class Curriculum(ABC):
    """Abstract curriculum interface.

    Provides a sequence of stages that inner-loop training will iterate over.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def stages(self) -> Iterable[Any]:
        """Yield curriculum stages (task-dependent type)."""
        ...
