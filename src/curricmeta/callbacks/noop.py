from __future__ import annotations
from typing import Any, Mapping, Optional

from curricmeta.callbacks.base import Callback
from curricmeta.utils.registry import register


@register("callbacks", "noop")
class NoOpCallback(Callback):
    """Default do-nothing callback."""

    def __init__(self, config: Optional[Mapping[str, Any]] = None):
        super().__init__(config=config)
