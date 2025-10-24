import os

# Flag to tell the registry not to auto-rebuild recursively
os.environ["COMPEHNDLY_BUILDING_REGISTRY"] = "1"

from compehndly.core.registry import build_manifest

if __name__ == "__main__":
    build_manifest()
