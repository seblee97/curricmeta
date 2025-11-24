from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict


class Model(ABC):
    """Abstract model interface.

    Concrete models will likely wrap a torch.nn.Module (or JAX / other),
    but we keep it generic at this skeleton stage.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def initialize(self, rng: Any | None = None) -> None:
        """Initialize trainable parameters / state."""
        ...
