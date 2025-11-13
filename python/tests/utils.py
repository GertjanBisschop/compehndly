import pyarrow as pa
import pyarrow.compute as pc

__registrations__ = []


def register(registry_name, name, version):
    def decorator(func):
        __registrations__.append((registry_name, name, version, func))
        return func

    return decorator


@register(registry_name="default", name="add_one", version="0.0.1")
def _add_one_v0_0_1(x: int):
    return x + 1


@register(registry_name="default", name="add_one", version="0.1.0")
def _add_one_v0_1_0(x):
    if x > 0:
        return x + 1
    else:
        return 0


@register(registry_name="default", name="normalize", version="0.0.1")
def _normalize_v0_0_1(biomarker: pa.Array, sg: pa.Array, *, sg_ref: float = 2.0) -> pa.Array:
    sg_minus_1 = pc.subtract(sg, pa.scalar(1.0))
    numerator = pc.multiply(biomarker, pa.scalar(sg_ref - 1.0))
    non_zero_mask = pc.not_equal(sg_minus_1, pa.scalar(0.0))
    safe_denominator = pc.if_else(non_zero_mask, sg_minus_1, None)
    raw_result = pc.divide(numerator, safe_denominator)

    return raw_result
