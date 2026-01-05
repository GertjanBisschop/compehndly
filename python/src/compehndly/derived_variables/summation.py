import pyarrow as pa
import pyarrow.compute as pc


__registrations__ = []


# TODO: move decorator for joint use
def register(registry_name, name, version):
    def decorator(func):
        __registrations__.append((registry_name, name, version, func))
        return func

    return decorator


def _summation_v0_0_1_reference(
    *arrays: float,
) -> float:
    if not arrays:
        raise ValueError("At least one input array is required")

    ret = 0
    for val in arrays:
        if val is not None:
            ret += val

    return ret


@register(registry_name="default", name="summation", version="0.0.1")
def _summation_v0_0_1_arrow(*arrays: pa.Array) -> pa.Array:
    """
    Vectorized summation over multiple Arrow arrays.

    Semantics:
    - If ANY input array is entirely null, return an all-null array.
    - Otherwise, nulls are treated as zero during summation.
    """

    if not arrays:
        raise ValueError("At least one input array is required")

    length = len(arrays[0])

    for arr in arrays:
        if len(arr) != length:
            raise ValueError("All input arrays must have the same length")

        if arr.null_count == length:
            return pa.nulls(length, type=pa.float64())

    result = pc.fill_null(arrays[0], 0)

    for arr in arrays[1:]:
        result = pc.add(result, pc.fill_null(arr, 0))

    return result
