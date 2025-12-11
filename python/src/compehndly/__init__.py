import functools

from compehndly.core.registry import FunctionRegistry
from compehndly.adapters import register_all_adapters

register_all_adapters()

# Internal override hook
_REGISTRY_BUILDER = None


def _set_registry_builder(builder):
    """
    Test helper: override how the registry is built.
    `builder` must be a callable returning a FunctionRegistry instance.
    """
    global _REGISTRY_BUILDER
    _REGISTRY_BUILDER = builder
    _REGISTRY.cache_clear()  # ensures next call uses new builder


@functools.lru_cache
def _REGISTRY():
    """
    Lazily build the registry, using the override when present.
    """
    if _REGISTRY_BUILDER is not None:
        return _REGISTRY_BUILDER()

    return FunctionRegistry.build_registry()


class _FunctionAccessor:
    """
    Returned from compehndly.<func_name>
    and provides:
        compehndly.add_one(...)
        compehndly.add_one["0.0.1"](...)
    """

    def __init__(self, name: str):
        self._name = name

    def __call__(self, *args, **kwargs):
        """Invoke latest version."""
        registry = _REGISTRY()
        fn = registry.get(self._name, version=None)
        return fn(*args, **kwargs)

    def __getitem__(self, version: str | None):
        """Invoke specific version (or None for latest)."""
        registry = _REGISTRY()
        fn = registry.get(self._name, version)
        return fn


def __getattr__(name: str):
    """
    Dynamically expose registered functions as attributes:
        compehndly.add_one
    """
    registry = _REGISTRY()

    if name in registry._functions:
        return _FunctionAccessor(name)

    raise AttributeError(f"'compehndly' has no function '{name}'")


__all__ = ["FunctionRegistry"]
