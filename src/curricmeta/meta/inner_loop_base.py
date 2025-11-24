from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict


class InnerLoop(ABC):
    """Inner-loop learner.

    Given a task, model, and meta-parameters, defines how to run
    curriculum-based training.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def run(self, task: Any, model: Any, meta_params: Dict[str, Any]) -> Dict[str, Any]:
        """Run inner-loop training; return artifacts for the outer loop."""
        ...
