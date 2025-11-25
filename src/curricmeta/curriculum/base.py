from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List


class Curriculum(ABC):
    """
    Task-agnostic curriculum interface.

    Given a number of stages (e.g. difficulty levels) exposed by a task,
    the curriculum returns a *schedule* over stage indices.

    The task handles sampling; the curriculum only decides which stage
    index to present when.
    """

    @abstractmethod
    def build_schedule(self, num_stages: int) -> List[int]:
        """
        Return a list of stage indices (0..num_stages-1) specifying the
        order in which stages will be visited during inner-loop training.
        """
        ...
