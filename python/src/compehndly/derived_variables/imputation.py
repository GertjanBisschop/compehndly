import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from compehndly.derived_variables.statsutils import fit_censored_lognorm

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


def _random_single_imputation_reference_v0_0_1(
    measurement: float,
    loq: float,
    lod: float | None = None,
) -> float:
    # TODO:
    pass


@register(registry_name="default", name="random_single_imputation", version="0.0.1")
def _random_single_imputation_arrow_v0_0_1(
    biomarker_pa: pa.Array,
    lod: float,
    loq: float,
    seed: int | None = None,
) -> pa.Array:
    """
    Perform random single imputation for left-censored lognormal data
    using PyArrow arrays for maximum compatibility.

    biomarker_pa : arrow array of floats or censored indicators (-1, -2, -3)
    lod       : limit of detection
    loq       : limit of quantification
    """

    # Convert Arrow → NumPy
    # NOTE: suboptimal solution
    biomarker = biomarker_pa.to_numpy(zero_copy_only=False)

    # Fill NA as -1 (your original convention)
    is_na = np.isnan(biomarker)
    biomarker_filled = np.where(is_na, -1, biomarker)

    # Censored if negative category code
    censored = biomarker_filled < 0
    values_np = np.where(censored, lod, biomarker_filled)
    dist = fit_censored_lognorm(values_np, censored)
    rng = np.random.default_rng(seed=seed)

    # Compute sampling bounds (vectorized)
    lower = np.zeros_like(biomarker_filled, dtype=float)
    upper = np.zeros_like(biomarker_filled, dtype=float)

    cat_below_lod = biomarker_filled == -1
    cat_between = biomarker_filled == -2
    cat_below_loq = biomarker_filled == -3

    # <LOD -> [0, LOD]
    lower[cat_below_lod] = 0
    upper[cat_below_lod] = lod
    # Between LOD & LOQ -> [LOD, LOQ]
    lower[cat_between] = lod
    upper[cat_between] = loq

    # <LOQ -> [0, LOQ]
    lower[cat_below_loq] = 0
    upper[cat_below_loq] = loq

    # Convert bounds to CDF space
    cdf_lo = dist.cdf(lower)
    cdf_hi = dist.cdf(upper)

    # generate U ~ Uniform(cdf_lo, cdf_hi)
    u = rng.uniform(cdf_lo, cdf_hi)
    imputed = dist.ppf(u)
    # replace censored with imputed
    result = biomarker.copy()
    result[censored] = imputed[censored]

    return pa.array(result)
