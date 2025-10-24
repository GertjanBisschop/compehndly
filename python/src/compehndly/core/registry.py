import json
import importlib
import os
import pkgutil

import compehndly

from pathlib import Path
from packaging.version import Version

# Global manifest (in-memory) used during development / build
_MANIFEST = {}


def get_registry():
    """Return a registry; rebuild manifest if missing."""
    manifest_path = Path(__file__).parent / "manifest.json"
    try:
        return LazyFunctionRegistry(manifest_path)
    except (FileNotFoundError, json.JSONDecodeError):
        # If explicitly building, don’t rebuild again!
        if os.environ.get("COMPEHNDLY_BUILDING_REGISTRY") == "1":
            raise
        print("[Registry] Manifest missing or invalid → rebuilding...")
        build_manifest()
        return LazyFunctionRegistry(manifest_path)


def register_function(name: str, version: str):
    """
    Returns a decorator that registers a function under `name`/`version`.
    """

    def decorator(func):
        module = func.__module__
        func_name = func.__name__
        _MANIFEST.setdefault(name, {})[version] = {
            "module": module,
            "function": func_name,
        }
        return func

    return decorator


def walk_function_modules():
    """Dynamically import all registry.functions.* modules to populate _MANIFEST."""

    package = compehndly
    for module_info in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        importlib.import_module(module_info.name)


def export_manifest(path=None):
    """
    Export the current in-memory manifest to JSON (for FAIR and lazy loading).
    Run this in a build step.
    """
    path = Path(path or Path(__file__).parent / "registry.json")
    with open(path, "w") as f:
        json.dump(_MANIFEST, f, indent=2, sort_keys=True)
    print(f"Exported manifest with {sum(len(v) for v in _MANIFEST.values())} entries to {path}")


def build_manifest():
    """
    Force a manifest rebuild by scanning all function modules.
    """
    print("[Registry] Building manifest...")
    walk_function_modules()
    export_manifest()
    print("[Registry] Manifest build complete.")


class LazyFunctionRegistry:
    """
    Loads function metadata from manifest.json and lazy-loads functions on demand.
    """

    def __init__(self, manifest_path=None):
        manifest_path = manifest_path or Path(__file__).parent / "registry.json"
        with open(manifest_path) as f:
            self.manifest = json.load(f)
        self._cache = {}

    def get(self, name, version=None):
        """Get function by name/version, lazy-importing the module if necessary."""
        if name not in self.manifest:
            raise KeyError(f"Function '{name}' not found in manifest.")

        versions = sorted(self.manifest[name].keys(), key=Version)
        version = str(version or versions[-1])
        cache_key = (name, version)

        if cache_key not in self._cache:
            entry = self.manifest[name][version]
            module = importlib.import_module(entry["module"])
            func = getattr(module, entry["function"])
            self._cache[cache_key] = func

        return self._cache[cache_key]

    def latest(self, name):
        return self.get(name)

    def list_versions(self, name):
        return sorted(self.manifest.get(name, {}).keys(), key=Version)


# Optional global instance (only when not in build mode)
if os.environ.get("COMPEHNDLY_BUILDING_REGISTRY") != "1":
    registry = get_registry()
else:
    registry = None
