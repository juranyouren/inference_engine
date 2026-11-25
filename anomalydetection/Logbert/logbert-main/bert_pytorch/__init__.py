"""LogBERT public classes, loaded when requested."""
from importlib import import_module

_MODULES = {"BERT": ".model", "Trainer": ".train_log", "Predictor": ".predict_log"}
__all__ = list(_MODULES)


def __getattr__(name):
    if name not in _MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_MODULES[name], __name__), name)
    globals()[name] = value
    return value
