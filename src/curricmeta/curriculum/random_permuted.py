from __future__ import annotations

from typing import Any, Dict, List
import random

from curricmeta.curriculum.base import Curriculum
from curricmeta.utils.registry import register


@register("curriculum", "random_permuted")
class RandomPermutedCurriculum(Curriculum):
    """
    Curriculum that shuffles the order of stages for each pass.
    """

    def __init__(self, config: Dict[str, Any]):
        self.passes: int = int(config.get("passes", 1))
        self.seed: int | None = config.get("seed", None)

    def build_schedule(self, num_stages: int) -> List[int]:
        rng = random.Random(self.seed)
        schedule: List[int] = []
        for _ in range(self.passes):
            indices = list(range(num_stages))
            rng.shuffle(indices)
            schedule.extend(indices)
        return schedule
