from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict


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
