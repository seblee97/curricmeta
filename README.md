# curricmeta

Skeleton codebase for experiment-registry centric meta-learning + curriculum learning experiments.

This is just a framework skeleton:
- Hydra config structure
- Registries for experiments, tasks, models, inner/outer loops, curricula, callbacks
- Callback system
- A dummy experiment wired end-to-end

You can run the dummy experiment (once installed / on PYTHONPATH) via:

```bash
python -m curricmeta.scripts.train experiment=dummy_experiment
```
