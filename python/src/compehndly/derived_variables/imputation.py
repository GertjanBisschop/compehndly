import pyarrow as pa
import pyarrow.compute as pc

__registrations__ = []


# TODO: move decorator for joint use
def register(registry_name, name, version):
    def decorator(func):
        __registrations__.append((registry_name, name, version, func))
        return func

    return decorator


def _medium_bound_imputation_v0_0_1_reference(
    measurement: float,
    loq: float,
    lod: float | None = None,
) -> float:
    if lod is not None:
        assert lod > 0.0
    assert loq > 0.0
    ret = measurement

    if lod is None:
        if measurement < loq:
            ret = loq / 2
    else:
        assert lod < loq
        if measurement < lod:
            ret = lod / 2
        elif measurement < loq:
            ret = (loq + lod) / 2

    return ret


@register(registry_name="default", name="medium_bound_imputation", version="0.0.1")
def _medium_bound_imputation_v0_0_1_arrow(
    measurement: pa.Array,
    loq: float,
    lod: float | None = None,
) -> pa.Array:
    """
    Vectorized medium-bound imputation.

    """

    if loq <= 0:
        raise ValueError("loq must be > 0")

    if lod is not None:
        if lod <= 0:
            raise ValueError("lod must be > 0")
        if lod >= loq:
            raise ValueError("lod must be < loq")

    # Start with identity (original measurement)
    result = measurement

    if lod is None:
        # measurement < loq → loq / 2
        mask = pc.less(measurement, loq)
        result = pc.if_else(mask, loq / 2, result)

    else:
        # measurement < lod → lod / 2
        mask_below_lod = pc.less(measurement, lod)
        result = pc.if_else(mask_below_lod, lod / 2, result)

        # lod <= measurement < loq → (lod + loq) / 2
        mask_between = pc.and_(
            pc.greater_equal(measurement, lod),
            pc.less(measurement, loq),
        )
        result = pc.if_else(mask_between, (lod + loq) / 2, result)

    return result


@register(registry_name="default", name="medium_bound_imputation_array", version="0.0.1")
def _medium_bound_imputation_v0_0_1_arrow_array(
    measurement: pa.Array,
    loq: pa.Array,
    lod: pa.Array | None = None,
) -> pa.Array:
    """
    Vectorized medium-bound imputation with row-wise LOQ / LOD arrays.
    """

    length = len(measurement)

    if len(loq) != length:
        raise ValueError("measurement and loq must have the same length")

    if lod is not None and len(lod) != length:
        raise ValueError("measurement and lod must have the same length")

    result = measurement

    if lod is None:
        # measurement < loq → loq / 2
        mask = pc.less(measurement, loq)
        result = pc.if_else(mask, pc.divide(loq, 2.0), result)

    else:
        # measurement < lod → lod / 2
        mask_below_lod = pc.less(measurement, lod)
        result = pc.if_else(mask_below_lod, pc.divide(lod, 2.0), result)

        # lod <= measurement < loq → (lod + loq) / 2
        mask_between = pc.and_(
            pc.greater_equal(measurement, lod),
            pc.less(measurement, loq),
        )
        midpoint = pc.divide(pc.add(lod, loq), 2.0)
        result = pc.if_else(mask_between, midpoint, result)

    return result


def _log_norm_imputation_v0_0_1_reference(
    measurement: float,
    loq: float,
    lod: float | None = None,
) -> float:
    pass


@register(registry_name="default", name="log_norm_imputation", version="0.0.1")
def _log_norm_imputation_v0_0_1():
    pass
