"""LUVE's integrated inference model and pipeline (lazy public imports)."""
from importlib import import_module
from typing import Any

__version__ = "0.3.0"
_EXPORTS = {
    "LuveConfig": ("configuration", "LuveConfig"),
    "FrequencyContext": ("modeling", "FrequencyContext"),
    "FrequencyProjection": ("modeling", "FrequencyProjection"),
    "LuveDiT": ("modeling", "LuveDiT"),
    "load_pretrained": ("checkpoint", "load_pretrained"),
    "LUVEPipeline": ("pipeline", "LUVEPipeline"),
}
__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve public symbols on demand, keeping CLI imports free of torch."""
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(f"{__name__}.{module}"), attribute)
    globals()[name] = value
    return value
