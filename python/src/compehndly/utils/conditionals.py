import pyarrow as pa
import pyarrow.compute as pc

__registrations__ = []


# TODO: move decorator for joint use
def register(registry_name, name, version):
    def decorator(func):
        __registrations__.append((registry_name, name, version, func))
        return func

    return decorator


# TODO: add conditional helper functions
