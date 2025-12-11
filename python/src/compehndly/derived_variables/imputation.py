import pyarrow as pa
import pyarrow.compute as pc

__registrations__ = []


# TODO: move decorator for joint use
def register(registry_name, name, version):
    def decorator(func):
        __registrations__.append((registry_name, name, version, func))
        return func

    return decorator


# we need to think here about some general code implementation
# a general code dict to be passed to any function


def _medium_bound_imputation_v0_0_1_reference(measurement: float, loq: float, lod: float | None = None) -> float:
    assert lod > 0.0
    assert loq > 0.0
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
def _medium_bound_imputation_v0_0_1():
    pass
