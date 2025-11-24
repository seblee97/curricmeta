REGISTRY = {
    "experiment": {},
    "task": {},
    "model": {},
    "inner_loop": {},
    "outer_loop": {},
    "curriculum": {},
    "optimizer": {},
    "callbacks": {},
    "metric": {},
}


def register(kind: str, name: str):
    """
    Decorator to register a class/function under REGISTRY[kind][name].

    Usage:
        @register("model", "mlp")
        class MLP(...): ...
    """
    def deco(obj):
        REGISTRY[kind][name] = obj
        return obj

    return deco


def build(kind: str, name: str, **kwargs):
    """
    Look up a component and instantiate/call it with kwargs.

    This is exactly your rlcap.build logic:
    - if stored object is callable, call it with kwargs
    - otherwise just return it
    """
    if name not in REGISTRY[kind]:
        raise KeyError(
            f"{kind} '{name}' not registered; have: {list(REGISTRY[kind].keys())}"
        )
    f = REGISTRY[kind][name]
    return f(**kwargs) if callable(f) else f
