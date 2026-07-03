from __future__ import annotations

import numpy as np
import polars as pl

from compehndly.derived_variables.statsutils import fit_censored_lognorm
from compehndly.polars.kernels import DerivedFunctionSpec


def _validate_scalar_thresholds(
    loq: float | None = None,
    lod: float | None = None,
) -> None:
    if loq is None and lod is None:
        raise ValueError("at least one of lod or loq must be provided")

    if loq is not None and loq <= 0:
        raise ValueError("loq must be > 0")

    if lod is not None:
        if lod <= 0:
            raise ValueError("lod must be > 0")
        if loq is not None and lod > loq:
            raise ValueError("lod must be <= loq")


def _threshold_array(value: float | None, length: int) -> np.ndarray:
    fill_value = np.nan if value is None else value
    return np.full(length, fill_value, dtype=float)


def _null_mask(series: pl.Series) -> np.ndarray:
    return series.is_null().to_numpy()


def _float_series_with_nulls(
    name: str | None,
    values: np.ndarray,
    null_mask: np.ndarray | None = None,
    *,
    nan_as_null: bool = True,
) -> pl.Series:
    if null_mask is None:
        null_mask = np.zeros(values.size, dtype=bool)

    output = [
        None if is_null or (nan_as_null and np.isnan(value)) else float(value)
        for value, is_null in zip(values, null_mask, strict=True)
    ]
    return pl.Series(name=name, values=output, dtype=pl.Float64)


def _bool_series_with_nulls(
    name: str | None,
    values: np.ndarray,
    null_mask: np.ndarray,
) -> pl.Series:
    output = [
        None if is_null else bool(value)
        for value, is_null in zip(values, null_mask, strict=True)
    ]
    return pl.Series(name=name, values=output, dtype=pl.Boolean)


def _expr_missing(expr: pl.Expr) -> pl.Expr:
    return expr.is_null() | expr.is_nan().fill_null(False)


def _expr_present(expr: pl.Expr) -> pl.Expr:
    return ~_expr_missing(expr)


def _validate_threshold_arrays(lod_np: np.ndarray, loq_np: np.ndarray) -> None:
    invalid_thresholds = (
        (~np.isnan(lod_np) & (lod_np <= 0))
        | (~np.isnan(loq_np) & (loq_np <= 0))
        | (~np.isnan(lod_np) & ~np.isnan(loq_np) & (lod_np >= loq_np))
    )
    if np.any(invalid_thresholds):
        raise ValueError("lod values must be > 0 and < loq values")


def _parse_bin_decoding_pairs(
    kwargs: dict[str, object],
) -> list[tuple[int, float, pl.Series | pl.Expr]]:
    filter_value_by_index: dict[int, float] = {}
    copy_from_by_index: dict[int, pl.Series | pl.Expr] = {}
    invalid_names: list[str] = []

    for name, value in kwargs.items():
        if name.startswith("filter_value_"):
            suffix = name.removeprefix("filter_value_")
            if not suffix.isdigit():
                invalid_names.append(name)
                continue
            try:
                filter_value_by_index[int(suffix)] = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be a float") from exc
        elif name.startswith("copy_from_"):
            suffix = name.removeprefix("copy_from_")
            if not suffix.isdigit():
                invalid_names.append(name)
                continue
            if not isinstance(value, (pl.Series, pl.Expr)):
                raise TypeError(f"{name} must be a Polars Series or Expr")
            copy_from_by_index[int(suffix)] = value
        else:
            invalid_names.append(name)

    if invalid_names:
        names = ", ".join(sorted(invalid_names))
        raise ValueError(
            "Unexpected arguments for bin_decoding: "
            f"{names}. Use filter_value_N/copy_from_N pairs."
        )

    indices = set(filter_value_by_index) | set(copy_from_by_index)
    if not indices:
        raise ValueError(
            "At least one filter_value_N/copy_from_N pair is required"
        )

    missing_filter_values = sorted(indices - set(filter_value_by_index))
    missing_copy_from = sorted(indices - set(copy_from_by_index))
    if missing_filter_values or missing_copy_from:
        message_parts = []
        if missing_filter_values:
            message_parts.append(
                "missing filter_value_"
                + ", filter_value_".join(
                    str(index) for index in missing_filter_values
                )
            )
        if missing_copy_from:
            message_parts.append(
                "missing copy_from_"
                + ", copy_from_".join(
                    str(index) for index in missing_copy_from
                )
            )
        raise ValueError("; ".join(message_parts))

    expected_indices = set(range(1, max(indices) + 1))
    if indices != expected_indices:
        missing_indices = sorted(expected_indices - indices)
        raise ValueError(
            "filter_value_N/copy_from_N indices must start at 1 and be "
            "contiguous; missing indices: "
            + ", ".join(str(index) for index in missing_indices)
        )

    filter_values = list(filter_value_by_index.values())
    non_nan_filter_values = [
        filter_value
        for filter_value in filter_values
        if not np.isnan(filter_value)
    ]
    if len(non_nan_filter_values) != len(set(non_nan_filter_values)):
        raise ValueError("filter_value_N values must be unique")
    if sum(np.isnan(filter_value) for filter_value in filter_values) > 1:
        raise ValueError("filter_value_N values must be unique")

    return [
        (
            index,
            filter_value_by_index[index],
            copy_from_by_index[index],
        )
        for index in sorted(indices)
    ]


def lab_sensitivity_dichotomization_kernel(
    measurement: pl.Series,
    loq: pl.Series | None = None,
    lod: pl.Series | None = None,
) -> pl.Series:
    length = len(measurement)
    if loq is None and lod is None:
        raise ValueError("at least one of lod or loq must be provided")
    if loq is not None and len(loq) != length:
        raise ValueError("measurement and loq must have the same length")
    if lod is not None and len(lod) != length:
        raise ValueError("measurement and lod must have the same length")

    measurement_np = measurement.cast(pl.Float64).to_numpy()
    measurement_null = _null_mask(measurement)
    loq_np = (
        np.full(length, np.nan, dtype=float)
        if loq is None
        else loq.cast(pl.Float64).to_numpy()
    )
    lod_np = (
        np.full(length, np.nan, dtype=float)
        if lod is None
        else lod.cast(pl.Float64).to_numpy()
    )

    has_lod = ~np.isnan(lod_np)
    has_loq = ~np.isnan(loq_np)
    threshold = np.where(has_lod, lod_np, loq_np)
    missing_result = measurement_null | (~has_lod & ~has_loq)
    result = measurement_np < threshold

    return _bool_series_with_nulls(
        measurement.name,
        result,
        missing_result,
    )


def lab_sensitivity_dichotomization_expr(
    measurement: pl.Expr,
    loq: pl.Expr | None = None,
    lod: pl.Expr | None = None,
) -> pl.Expr:
    if loq is None and lod is None:
        raise ValueError("at least one of lod or loq must be provided")

    measurement_missing = _expr_missing(measurement)
    if lod is None:
        loq_missing = _expr_missing(loq)
        return (
            pl.when(measurement_missing | loq_missing)
            .then(None)
            .when(measurement < loq)
            .then(True)
            .otherwise(False)
        )
    if loq is None:
        lod_missing = _expr_missing(lod)
        return (
            pl.when(measurement_missing | lod_missing)
            .then(None)
            .when(measurement < lod)
            .then(True)
            .otherwise(False)
        )

    lod_present = _expr_present(lod)
    loq_present = _expr_present(loq)
    threshold = pl.when(lod_present).then(lod).otherwise(loq)
    return (
        pl.when(measurement_missing | (~lod_present & ~loq_present))
        .then(None)
        .when(measurement < threshold)
        .then(True)
        .otherwise(False)
    )


def medium_bound_imputation_scalar_input_kernel(
    measurement: pl.Series,
    loq: float | None = None,
    lod: float | None = None,
) -> pl.Series:
    _validate_scalar_thresholds(loq, lod)

    measurement_np = measurement.cast(pl.Float64).to_numpy()
    measurement_null = _null_mask(measurement)
    result = measurement_np.copy()

    if lod is None:
        mask = (measurement_np < loq) & ~np.isnan(measurement_np)
        result[mask] = loq / 2
        return _float_series_with_nulls(
            measurement.name, result, measurement_null
        )

    if loq is None:
        mask = (measurement_np < lod) & ~np.isnan(measurement_np)
        result[mask] = lod / 2
        return _float_series_with_nulls(
            measurement.name, result, measurement_null
        )

    mask_below_lod = (measurement_np < lod) & ~np.isnan(measurement_np)
    result[mask_below_lod] = lod / 2

    midpoint = (lod + loq) / 2
    mask_between = (
        (measurement_np >= lod)
        & (measurement_np < loq)
        & ~np.isnan(measurement_np)
    )
    result[mask_between] = midpoint
    return _float_series_with_nulls(measurement.name, result, measurement_null)


def medium_bound_imputation_scalar_input_expr(
    measurement: pl.Expr,
    loq: float | None = None,
    lod: float | None = None,
) -> pl.Expr:
    _validate_scalar_thresholds(loq, lod)

    result = measurement
    if lod is None:
        return pl.when(measurement < loq).then(loq / 2).otherwise(result)
    if loq is None:
        return pl.when(measurement < lod).then(lod / 2).otherwise(result)

    result = pl.when(measurement < lod).then(lod / 2).otherwise(result)
    midpoint = (lod + loq) / 2
    return (
        pl.when((measurement >= lod) & (measurement < loq))
        .then(midpoint)
        .otherwise(result)
    )


def medium_bound_imputation_kernel(
    measurement: pl.Series,
    loq: pl.Series | None = None,
    lod: pl.Series | None = None,
) -> pl.Series:
    length = len(measurement)
    if loq is None and lod is None:
        raise ValueError("at least one of lod or loq must be provided")
    if loq is not None and len(loq) != length:
        raise ValueError("measurement and loq must have the same length")
    if lod is not None and len(lod) != length:
        raise ValueError("measurement and lod must have the same length")

    measurement_np = measurement.cast(pl.Float64).to_numpy()
    measurement_null = _null_mask(measurement)
    loq_np = (
        np.full(length, np.nan, dtype=float)
        if loq is None
        else loq.cast(pl.Float64).to_numpy()
    )
    result = measurement_np.copy()

    if lod is None:
        mask = (
            (measurement_np < loq_np)
            & ~np.isnan(measurement_np)
            & ~np.isnan(loq_np)
        )
        result[mask] = loq_np[mask] / 2
        result_null = measurement_null | np.isnan(loq_np)
        return _float_series_with_nulls(measurement.name, result, result_null)

    lod_np = lod.cast(pl.Float64).to_numpy()

    if loq is None:
        mask = (
            (measurement_np < lod_np)
            & ~np.isnan(measurement_np)
            & ~np.isnan(lod_np)
        )
        result[mask] = lod_np[mask] / 2
        result_null = measurement_null | np.isnan(lod_np)
        return _float_series_with_nulls(measurement.name, result, result_null)

    mask_below_lod = (
        (measurement_np < lod_np)
        & ~np.isnan(measurement_np)
        & ~np.isnan(lod_np)
    )
    result[mask_below_lod] = lod_np[mask_below_lod] / 2

    mask_lod_missing_loq_present = (
        (measurement_np < loq_np)
        & ~np.isnan(measurement_np)
        & np.isnan(lod_np)
        & ~np.isnan(loq_np)
    )
    result[mask_lod_missing_loq_present] = (
        loq_np[mask_lod_missing_loq_present] / 2
    )

    midpoint = (lod_np + loq_np) / 2
    mask_between = (
        (measurement_np >= lod_np)
        & (measurement_np < loq_np)
        & ~np.isnan(measurement_np)
        & ~np.isnan(lod_np)
        & ~np.isnan(loq_np)
    )
    result[mask_between] = midpoint[mask_between]

    result_null = measurement_null | (np.isnan(lod_np) & np.isnan(loq_np))
    return _float_series_with_nulls(measurement.name, result, result_null)


def medium_bound_imputation_expr(
    measurement: pl.Expr,
    loq: pl.Expr | None = None,
    lod: pl.Expr | None = None,
) -> pl.Expr:
    if loq is None and lod is None:
        raise ValueError("at least one of lod or loq must be provided")

    result = measurement
    if lod is None:
        return (
            pl.when(_expr_missing(loq))
            .then(None)
            .when(measurement < loq)
            .then(loq / 2)
            .otherwise(result)
        )
    if loq is None:
        return (
            pl.when(_expr_missing(lod))
            .then(None)
            .when(measurement < lod)
            .then(lod / 2)
            .otherwise(result)
        )

    lod_missing = _expr_missing(lod)
    loq_missing = _expr_missing(loq)
    lod_present = ~lod_missing
    loq_present = ~loq_missing

    result = (
        pl.when(lod_present & (measurement < lod))
        .then(lod / 2)
        .otherwise(result)
    )
    result = (
        pl.when(lod_missing & loq_present & (measurement < loq))
        .then(loq / 2)
        .otherwise(result)
    )
    midpoint = (lod + loq) / 2
    return (
        pl.when(lod_missing & loq_missing)
        .then(None)
        .when(
            lod_present
            & loq_present
            & (measurement >= lod)
            & (measurement < loq)
        )
        .then(midpoint)
        .otherwise(result)
    )


def bin_decoding_kernel(
    *,
    values: pl.Series,
    **kwargs: object,
) -> pl.Series:
    """
    Replace sentinel values by copying from paired Series inputs.

    Call contract:
      values=<Series>,
      filter_value_1=<scalar>, copy_from_1=<Series>,
      filter_value_2=<scalar>, copy_from_2=<Series>,
      ...

    Rule indices must start at 1, be contiguous, form complete pairs, and
    use unique `filter_value_N` sentinel values. No other kwargs are accepted.
    """
    pairs = _parse_bin_decoding_pairs(kwargs)

    values_np = values.cast(pl.Float64).to_numpy()
    result_null = _null_mask(values)
    result = values_np.copy()

    for _, filter_value, copy_from in pairs:
        if len(copy_from) != len(values):
            raise ValueError(
                "values and copy_from_N inputs must have the same length"
            )

        copy_from_np = copy_from.cast(pl.Float64).to_numpy()
        copy_from_null = _null_mask(copy_from)
        if np.isnan(filter_value):
            mask = np.isnan(values_np) & ~result_null
        else:
            mask = (values_np == filter_value) & ~result_null
        result[mask] = copy_from_np[mask]
        result_null[mask] = copy_from_null[mask]

    return _float_series_with_nulls(
        values.name,
        result,
        result_null,
        nan_as_null=False,
    )


def bin_decoding_expr(
    *,
    values: pl.Expr,
    **kwargs: object,
) -> pl.Expr:
    """
    Replace sentinel values by copying from paired expression inputs.

    Call contract:
      values=<Expr>,
      filter_value_1=<scalar>, copy_from_1=<Expr>,
      filter_value_2=<scalar>, copy_from_2=<Expr>,
      ...

    Rule indices must start at 1, be contiguous, form complete pairs, and
    use unique `filter_value_N` sentinel values. No other kwargs are accepted.
    """
    pairs = _parse_bin_decoding_pairs(kwargs)

    result = values
    for _, filter_value, copy_from in pairs:
        if np.isnan(filter_value):
            mask = values.is_nan().fill_null(False)
        else:
            mask = values == filter_value
        result = pl.when(mask).then(copy_from).otherwise(result)

    return result


def _random_single_imputation_from_arrays(
    biomarker_np: np.ndarray,
    lod_np: np.ndarray,
    loq_np: np.ndarray,
    biomarker_null: np.ndarray | None = None,
    min_unique_values: int = 0,
    min_observed_percentage: int = 0,
    seed: int | None = None,
) -> np.ndarray:
    if biomarker_null is None:
        biomarker_null = np.zeros(biomarker_np.size, dtype=bool)

    biomarker_filled = biomarker_np.copy()
    sentinel = (biomarker_filled < 0) & ~biomarker_null

    cat_below_lod = sentinel & (biomarker_filled == -1)
    cat_between = sentinel & (biomarker_filled == -2)
    cat_below_loq = sentinel & (biomarker_filled == -3)
    invalid_sentinel = sentinel & ~(
        cat_below_lod | cat_between | cat_below_loq
    )

    missing_boundary = (
        (cat_below_lod & np.isnan(lod_np))
        | (cat_between & (np.isnan(lod_np) | np.isnan(loq_np)))
        | (cat_below_loq & np.isnan(loq_np))
    )
    censored = sentinel & ~missing_boundary & ~invalid_sentinel
    observed = ~biomarker_null & ~sentinel & ~np.isnan(biomarker_filled)

    # perform configurable data checks
    checks_failed = False

    # check: at least [min_observed_percentage] % of the values are above LOD/LOQ
    if not checks_failed:
        threshold = np.where(~np.isnan(lod_np), lod_np, loq_np)
        count_above_lod_loq = np.count_nonzero(
            observed & ~np.isnan(threshold) & (biomarker_filled > threshold)
        )
        denominator = np.count_nonzero(~biomarker_null)
        if (
            denominator == 0
            or count_above_lod_loq
            < denominator / 100.0 * min_observed_percentage
        ):
            checks_failed = True

    # check: at least [min_unique_values] unique values are observed above LOD/LOQ
    if not checks_failed:
        threshold = np.where(~np.isnan(lod_np), lod_np, loq_np)
        above_lod_loq = (
            observed & ~np.isnan(threshold) & (biomarker_filled > threshold)
        )
        count_unique_values_above_lod_loq = np.unique(
            biomarker_filled[above_lod_loq]
        ).size
        if count_unique_values_above_lod_loq < min_unique_values:
            checks_failed = True

    if checks_failed:
        result = np.full(biomarker_np.size, np.nan)
    else:
        lower = np.zeros_like(biomarker_filled, dtype=float)
        upper = np.zeros_like(biomarker_filled, dtype=float)

        lower[cat_below_lod] = 0
        upper[cat_below_lod] = lod_np[cat_below_lod]

        lower[cat_between] = lod_np[cat_between]
        upper[cat_between] = loq_np[cat_between]

        lower[cat_below_loq] = 0
        upper[cat_below_loq] = loq_np[cat_below_loq]

        fit_values = np.where(censored, upper, biomarker_filled)
        fit_mask = observed | censored
        dist = fit_censored_lognorm(fit_values[fit_mask], censored[fit_mask])
        rng = np.random.default_rng(seed=seed)

        cdf_lo = dist.cdf(lower)
        cdf_hi = dist.cdf(upper)

        u = np.zeros_like(biomarker_filled, dtype=float)
        u[censored] = rng.uniform(cdf_lo[censored], cdf_hi[censored])
        imputed = np.full_like(biomarker_filled, np.nan, dtype=float)
        imputed[censored] = dist.ppf(u[censored])

        result = biomarker_filled.copy()
        result[censored] = imputed[censored]
        result[biomarker_null | missing_boundary | invalid_sentinel] = np.nan

    return result


def random_single_imputation_scalar_input_kernel(
    biomarker: pl.Series,
    lod: float | None = None,
    loq: float | None = None,
    min_unique_values: int = 0,
    min_observed_percentage: int = 0,
    seed: int | None = None,
) -> pl.Series:
    _validate_scalar_thresholds(loq, lod)

    biomarker_np = biomarker.cast(pl.Float64).to_numpy()
    biomarker_null = _null_mask(biomarker)
    result = _random_single_imputation_from_arrays(
        biomarker_np=biomarker_np,
        lod_np=_threshold_array(lod, biomarker_np.size),
        loq_np=_threshold_array(loq, biomarker_np.size),
        biomarker_null=biomarker_null,
        min_unique_values=min_unique_values,
        min_observed_percentage=min_observed_percentage,
        seed=seed,
    )

    return _float_series_with_nulls(biomarker.name, result)


def random_single_imputation_scalar_input_expr(
    biomarker: pl.Expr,
    lod: float | None = None,
    loq: float | None = None,
    min_unique_values: int = 0,
    min_observed_percentage: int = 0,
    seed: int | None = None,
) -> pl.Expr:
    _validate_scalar_thresholds(loq, lod)

    return pl.struct([biomarker.alias("_biomarker")]).map_batches(
        lambda s: random_single_imputation_scalar_input_kernel(
            s.struct.field("_biomarker"),
            lod=lod,
            loq=loq,
            min_unique_values=min_unique_values,
            min_observed_percentage=min_observed_percentage,
            seed=seed,
        ),
        return_dtype=pl.Float64,
    )


def random_single_imputation_kernel(
    biomarker: pl.Series,
    lod: pl.Series | None = None,
    loq: pl.Series | None = None,
    min_unique_values: int = 0,
    min_observed_percentage: int = 0,
    seed: int | None = None,
) -> pl.Series:
    length = len(biomarker)
    if lod is None and loq is None:
        raise ValueError("at least one of lod or loq must be provided")
    if lod is not None and len(lod) != length:
        raise ValueError("biomarker and lod must have the same length")
    if loq is not None and len(loq) != length:
        raise ValueError("biomarker and loq must have the same length")

    biomarker_np = biomarker.cast(pl.Float64).to_numpy()
    biomarker_null = _null_mask(biomarker)
    lod_np = (
        np.full(length, np.nan, dtype=float)
        if lod is None
        else lod.cast(pl.Float64).to_numpy()
    )
    loq_np = (
        np.full(length, np.nan, dtype=float)
        if loq is None
        else loq.cast(pl.Float64).to_numpy()
    )

    _validate_threshold_arrays(lod_np, loq_np)

    result = _random_single_imputation_from_arrays(
        biomarker_np=biomarker_np,
        lod_np=lod_np,
        loq_np=loq_np,
        biomarker_null=biomarker_null,
        min_unique_values=min_unique_values,
        min_observed_percentage=min_observed_percentage,
        seed=seed,
    )

    return _float_series_with_nulls(biomarker.name, result)


def random_single_imputation_expr(
    biomarker: pl.Expr,
    lod: pl.Expr | None = None,
    loq: pl.Expr | None = None,
    min_unique_values: int = 0,
    min_observed_percentage: int = 0,
    seed: int | None = None,
) -> pl.Expr:
    if lod is None and loq is None:
        raise ValueError("at least one of lod or loq must be provided")

    fields = [biomarker.alias("_biomarker")]
    if lod is not None:
        fields.append(lod.alias("_lod"))
    if loq is not None:
        fields.append(loq.alias("_loq"))

    return pl.struct(fields).map_batches(
        lambda s: random_single_imputation_kernel(
            biomarker=s.struct.field("_biomarker"),
            lod=s.struct.field("_lod") if lod is not None else None,
            loq=s.struct.field("_loq") if loq is not None else None,
            min_unique_values=min_unique_values,
            min_observed_percentage=min_observed_percentage,
            seed=seed,
        ),
        return_dtype=pl.Float64,
    )


FUNCTION_SPECS = [
    DerivedFunctionSpec(
        name="lab_sensitivity_dichotomization",
        kernel=lab_sensitivity_dichotomization_kernel,
        expr_builder=lab_sensitivity_dichotomization_expr,
    ),
    DerivedFunctionSpec(
        name="medium_bound_imputation_scalar_input",
        kernel=medium_bound_imputation_scalar_input_kernel,
        expr_builder=medium_bound_imputation_scalar_input_expr,
    ),
    DerivedFunctionSpec(
        name="medium_bound_imputation",
        kernel=medium_bound_imputation_kernel,
        expr_builder=medium_bound_imputation_expr,
    ),
    DerivedFunctionSpec(
        name="bin_decoding",
        kernel=bin_decoding_kernel,
        expr_builder=bin_decoding_expr,
    ),
    DerivedFunctionSpec(
        name="random_single_imputation_scalar_input",
        kernel=random_single_imputation_scalar_input_kernel,
        expr_builder=random_single_imputation_scalar_input_expr,
    ),
    DerivedFunctionSpec(
        name="random_single_imputation",
        kernel=random_single_imputation_kernel,
        expr_builder=random_single_imputation_expr,
    ),
]
