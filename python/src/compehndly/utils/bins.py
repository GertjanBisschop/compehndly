import pyarrow as pa
import pyarrow.compute as pc

__registrations__ = []


# TODO: move decorator for joint use
def register(registry_name, name, version):
    def decorator(func):
        __registrations__.append((registry_name, name, version, func))
        return func

    return decorator


def _simple_bin_v0_0_1_reference(
    *arrays: float,
) -> float:
    pass


@register(registry_name="default", name="summation", version="0.0.1")
def _simple_bin_v0_0_1_arrow(
    array: pa.Array,
) -> pa.Array:
    pass
