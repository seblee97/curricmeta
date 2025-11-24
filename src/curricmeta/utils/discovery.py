from __future__ import annotations

import importlib
import pkgutil
from typing import Iterable


def _import_submodules(package) -> None:
    """
    Import all submodules of a given package, so that registry decorators run.

    This is similar to what your old rlcap.import_all_components likely did.
    """
    for module_info in pkgutil.walk_packages(
        package.__path__, package.__name__ + "."
    ):
        importlib.import_module(module_info.name)


def import_all_components() -> None:
    """
    Import all component packages (tasks, models, meta, curricula, callbacks,
    experiments) so their @register_* decorators populate the registries.
    """
    from curricmeta import (
        tasks,
        models,
        meta,
        curriculum,
        callbacks,
        experiments,
    )

    for pkg in (tasks, models, meta, curriculum, callbacks, experiments):
        _import_submodules(pkg)
