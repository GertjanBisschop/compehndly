import pyarrow as pa
import pyarrow.compute as pc

__registrations__ = []


# TODO: move decorator for joint use
def register(registry_name, name, version):
    def decorator(func):
        __registrations__.append((registry_name, name, version, func))
        return func

    return decorator


# ---------------- URINARY BIOMARKERS ----------------------


def _standardize_creatinine_v0_0_1_reference(
    measured: float,
    crt: float,
) -> float:
    "measured in micrograms/L, crt in mg/dL"
    ret = 100 * measured / crt

    return ret


@register(registry_name="default", name="standardize_creatinine", version="0.0.1")
def _standardize_creatinine_v0_0_1_arrow(measured: pa.Array, crt: pa.Array) -> pa.Array:
    ret = pc.divide(pc.multiply(measured, 100), crt)

    return ret


def _normalize_specific_gravity_v0_0_1_reference(
    measured: float,
    sg_measured: float,
    sg_ref: float,
) -> float:
    ret = measured * (sg_ref - 1) / sg_measured

    return ret


@register(registry_name="default", name="normalize_specific_gravity", version="0.0.1")
def _normalize_specific_gravity_v0_0_1_arrow(measured: pa.Array, sg_measured: pa.Array, sg_ref: float) -> pa.Array:
    # Compute (sg_ref - 1) as a scalar
    sg_factor = pa.scalar(sg_ref - 1, type=pa.float64())

    # measured * (sg_ref - 1)
    numerator = pc.multiply(measured, sg_factor)

    # (measured * (sg_ref - 1)) / sg_measured
    ret = pc.divide(numerator, sg_measured)

    return ret


# ---------------- LIPID SOLUBLE BIOMARKERS IN BLOOD ----------------------


def _standardize_lipid_v0_0_1_reference(
    measured: float,
    lipid_value: float,
) -> float:
    ret = 100 * measured / lipid_value

    return ret


@register(registry_name="default", name="normalize_specific_gravity", version="0.0.1")
def _standardize_lipid_v0_0_1_arrow(measured: pa.Array, lipid_value: pa.Array) -> pa.Array:
    ret = pc.divide(
        pc.multiply(measured, 100),
        lipid_value,
    )

    return ret


# ---------------- LIPID SOLUBLE BIOMARKERS IN MILK ----------------------
