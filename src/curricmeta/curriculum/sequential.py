from __future__ import annotations

from typing import Any, Dict, List

from curricmeta.curriculum.base import Curriculum
from curricmeta.utils.registry import register


@register("curriculum", "sequential")
class SequentialCurriculum(Curriculum):
    """
    Simple curriculum: iterate stages in order 0..num_stages-1
    for a given number of passes.
    """

    def __init__(self, config: Dict[str, Any]):
        self.passes: int = int(config.get("passes", 1))

    def build_schedule(self, num_stages: int) -> List[int]:
        one_pass = list(range(num_stages))
        return one_pass * self.passes
