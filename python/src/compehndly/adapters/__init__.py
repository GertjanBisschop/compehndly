import logging

from importlib.metadata import entry_points
from compehndly.adapters.base import ArrayAdapter

logger = logging.getLogger(__name__)

_ADAPTERS = {}


def register_adapter(adapter: ArrayAdapter):
    _ADAPTERS[adapter.name] = adapter


def register_all_adapters():
    logger.debug("Registering adapters")
    for ep in entry_points(group="compehndly.adapters"):
        try:
            logger.debug(f"loading {ep}")
            AdapterClass = ep.load()
            register_adapter(AdapterClass())
        except ImportError as e:
            logger.warning(f"Failed to load {ep.name}: {e}")

    register_adapter(ArrayAdapter())
