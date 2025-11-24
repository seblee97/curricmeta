from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict


class OuterLoop(ABC):
    """Outer-loop meta-learner.

    Parameterizes and updates meta-parameters (learning rates, init, etc.).
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def initialize_meta_params(self) -> Dict[str, Any]:
        """Create initial meta-parameters."""
        ...

    @abstractmethod
    def step(self, inner_results: Dict[str, Any], meta_params: Dict[str, Any]) -> Dict[str, Any]:
        """Update meta-parameters given inner-loop results."""
        ...
