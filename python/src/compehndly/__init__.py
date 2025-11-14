from compehndly.core.registry import FunctionRegistry
from compehndly.adapters import register_all_adapters

register_all_adapters()

__all__ = ["FunctionRegistry"]
