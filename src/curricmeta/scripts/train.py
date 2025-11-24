from __future__ import annotations

import logging
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from curricmeta.utils.discovery import import_all_components
from curricmeta.utils.seed import set_seed
from curricmeta.utils.logging import setup_logging
from curricmeta.utils.registry import build
from curricmeta.callbacks.base import CallbackList

from typing import List, Dict, Optional

log = logging.getLogger(__name__)


def _build_callbacks(items: Optional[List | Dict]) -> List:
    """
    Build callbacks from the given config.

    Items can either be
        - a dictionary where keys are names of callbacks and values are the arguments.
        - a list dictionary where each dictionary contains a "name" key and a "kwargs" key.
    """
    if items is None:
        return []
    out = []
    if isinstance(items, dict):
        # covers case where callbacks are given from separate yaml files
        for k, v in items.items():
            out.append(build("callbacks", k, **(v or {})))
    elif isinstance(items, list):
        # covers case where callbacks is given as
        for it in items:
            out.append(build("callbacks", it["name"], **(it.get("kwargs") or {})))
    else:
        raise TypeError(...)
    return CallbackList(out)


@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # Hydra run directory
    hydra_cfg = HydraConfig.get()
    out_dir = hydra_cfg.run.dir

    OmegaConf.set_struct(cfg, False)
    setup_logging(level=cfg.logging.level)

    import_all_components()
    set_seed(cfg.seed)

    Path(cfg.experiment.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.logging.log_dir).mkdir(parents=True, exist_ok=True)

    items = OmegaConf.to_container(cfg.get("callbacks"), resolve=True) or []

    callbacks = _build_callbacks(items)

    # Build experiment/trainer from registry, like rlcap build("trainer", ...)
    exp_name = str(cfg.experiment.name)
    experiment = build("experiment", exp_name, cfg=cfg, callbacks=callbacks)

    results = experiment.run()
    log.info("Experiment finished. Results: %s", results)


if __name__ == "__main__":
    main()
