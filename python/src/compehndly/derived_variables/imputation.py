from __future__ import annotations

from typing import NamedTuple

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


def _validate_threshold_arrays(lod_np: np.ndarray, loq_np: np.ndarray) -> None:
    invalid_thresholds = (
        (~np.isnan(lod_np) & (lod_np <= 0))
        | (~np.isnan(loq_np) & (loq_np <= 0))
        | (~np.isnan(lod_np) & ~np.isnan(loq_np) & (lod_np >= loq_np))
    )
    if np.any(invalid_thresholds):
        raise ValueError("lod values must be > 0 and < loq values")


class _MeasurementMasks(NamedTuple):
    not_measured: np.ndarray
    below_lod: np.ndarray
    between_lod_loq: np.ndarray
    below_loq: np.ndarray
    decimal: np.ndarray
    invalid_negative: np.ndarray


class _CensoringMasks(NamedTuple):
    below_lod: np.ndarray
    between_lod_loq: np.ndarray
    below_loq: np.ndarray

    @property
    def any(self) -> np.ndarray:
        return self.below_lod | self.between_lod_loq | self.below_loq


def _measurement_categories(
    values: np.ndarray,
    null_mask: np.ndarray,
) -> _MeasurementMasks:
    present = ~null_mask & ~np.isnan(values)
    not_measured = present & (values == -10)
    below_lod = present & (values == -1)
    between_lod_loq = present & (values == -2)
    below_loq = present & (values == -3)
    known_code = not_measured | below_lod | between_lod_loq | below_loq
    decimal = present & (values >= 0)
    invalid_negative = present & (values < 0) & ~known_code
    return _MeasurementMasks(
        not_measured=not_measured,
        below_lod=below_lod,
        between_lod_loq=between_lod_loq,
        below_loq=below_loq,
        decimal=decimal,
        invalid_negative=invalid_negative,
    )


def _unsupported_code_mask(
    masks: _MeasurementMasks,
    has_lod: np.ndarray,
    has_loq: np.ndarray,
) -> np.ndarray:
    return (
        (masks.below_lod & ~has_lod)
        | (masks.between_lod_loq & ~(has_lod & has_loq))
        | (masks.below_loq & (has_lod | ~has_loq))
    )


def _result_null_mask(
    values: np.ndarray,
    null_mask: np.ndarray,
    has_any_limit: np.ndarray,
    masks: _MeasurementMasks,
    unsupported_code: np.ndarray,
) -> np.ndarray:
    return (
        null_mask
        | np.isnan(values)
        | ~has_any_limit
        | masks.not_measured
        | masks.invalid_negative
        | unsupported_code
    )


def _censoring_masks(
    values: np.ndarray,
    lod_np: np.ndarray,
    loq_np: np.ndarray,
    masks: _MeasurementMasks,
    has_lod: np.ndarray,
    has_loq: np.ndarray,
) -> _CensoringMasks:
    below_lod = (masks.below_lod & has_lod) | (
        masks.decimal & has_lod & (values < lod_np)
    )
    between_lod_loq = (masks.between_lod_loq & has_lod & has_loq) | (
        masks.decimal
        & has_lod
        & has_loq
        & (values >= lod_np)
        & (values < loq_np)
    )
    below_loq = (masks.below_loq & ~has_lod & has_loq) | (
        masks.decimal & ~has_lod & has_loq & (values < loq_np)
    )
    return _CensoringMasks(
        below_lod=below_lod,
        between_lod_loq=between_lod_loq,
        below_loq=below_loq,
    )


def _medium_bound_imputation_from_arrays(
    measurement_np: np.ndarray,
    lod_np: np.ndarray,
    loq_np: np.ndarray,
    measurement_null: np.ndarray,
) -> np.ndarray:
    has_lod = ~np.isnan(lod_np)
    has_loq = ~np.isnan(loq_np)
    has_any_limit = has_lod | has_loq
    masks = _measurement_categories(measurement_np, measurement_null)
    censored = _censoring_masks(
        measurement_np,
        lod_np,
        loq_np,
        masks,
        has_lod,
        has_loq,
    )

    result = measurement_np.copy()
    result[censored.below_lod] = lod_np[censored.below_lod] / 2
    result[censored.between_lod_loq] = (
        lod_np[censored.between_lod_loq] + loq_np[censored.between_lod_loq]
    ) / 2
    result[censored.below_loq] = loq_np[censored.below_loq] / 2

    unsupported_code = _unsupported_code_mask(masks, has_lod, has_loq)
    result[
        _result_null_mask(
            measurement_np,
            measurement_null,
            has_any_limit,
            masks,
            unsupported_code,
        )
    ] = np.nan
    return result


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
    has_any_limit = has_lod | has_loq
    masks = _measurement_categories(measurement_np, measurement_null)

    threshold = np.where(has_loq, loq_np, lod_np)
    result = masks.decimal & has_any_limit & (measurement_np >= threshold)
    unsupported_code = _unsupported_code_mask(masks, has_lod, has_loq)
    missing_result = _result_null_mask(
        measurement_np,
        measurement_null,
        has_any_limit,
        masks,
        unsupported_code,
    )

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

    fields = [measurement.alias("_measurement")]
    if lod is not None:
        fields.append(lod.alias("_lod"))
    if loq is not None:
        fields.append(loq.alias("_loq"))

    return pl.struct(fields).map_batches(
        lambda s: lab_sensitivity_dichotomization_kernel(
            measurement=s.struct.field("_measurement"),
            lod=s.struct.field("_lod") if lod is not None else None,
            loq=s.struct.field("_loq") if loq is not None else None,
        ),
        return_dtype=pl.Boolean,
    )


def medium_bound_imputation_scalar_input_kernel(
    measurement: pl.Series,
    loq: float | None = None,
    lod: float | None = None,
) -> pl.Series:
    _validate_scalar_thresholds(loq, lod)

    measurement_np = measurement.cast(pl.Float64).to_numpy()
    measurement_null = _null_mask(measurement)
    result = _medium_bound_imputation_from_arrays(
        measurement_np=measurement_np,
        lod_np=_threshold_array(lod, measurement_np.size),
        loq_np=_threshold_array(loq, measurement_np.size),
        measurement_null=measurement_null,
    )
    return _float_series_with_nulls(measurement.name, result)


def medium_bound_imputation_scalar_input_expr(
    measurement: pl.Expr,
    loq: float | None = None,
    lod: float | None = None,
) -> pl.Expr:
    _validate_scalar_thresholds(loq, lod)

    return pl.struct([measurement.alias("_measurement")]).map_batches(
        lambda s: medium_bound_imputation_scalar_input_kernel(
            s.struct.field("_measurement"),
            lod=lod,
            loq=loq,
        ),
        return_dtype=pl.Float64,
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
    lod_np = (
        np.full(length, np.nan, dtype=float)
        if lod is None
        else lod.cast(pl.Float64).to_numpy()
    )

    result = _medium_bound_imputation_from_arrays(
        measurement_np=measurement_np,
        lod_np=lod_np,
        loq_np=loq_np,
        measurement_null=measurement_null,
    )
    return _float_series_with_nulls(measurement.name, result)


def medium_bound_imputation_expr(
    measurement: pl.Expr,
    loq: pl.Expr | None = None,
    lod: pl.Expr | None = None,
) -> pl.Expr:
    if loq is None and lod is None:
        raise ValueError("at least one of lod or loq must be provided")

    fields = [measurement.alias("_measurement")]
    if lod is not None:
        fields.append(lod.alias("_lod"))
    if loq is not None:
        fields.append(loq.alias("_loq"))

    return pl.struct(fields).map_batches(
        lambda s: medium_bound_imputation_kernel(
            measurement=s.struct.field("_measurement"),
            lod=s.struct.field("_lod") if lod is not None else None,
            loq=s.struct.field("_loq") if loq is not None else None,
        ),
        return_dtype=pl.Float64,
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
    has_lod = ~np.isnan(lod_np)
    has_loq = ~np.isnan(loq_np)
    has_any_limit = has_lod | has_loq
    masks = _measurement_categories(biomarker_filled, biomarker_null)
    censored = _censoring_masks(
        biomarker_filled,
        lod_np,
        loq_np,
        masks,
        has_lod,
        has_loq,
    )
    censored_any = censored.any

    unsupported_code = _unsupported_code_mask(masks, has_lod, has_loq)
    result_null = _result_null_mask(
        biomarker_filled,
        biomarker_null,
        has_any_limit,
        masks,
        unsupported_code,
    )
    observed = masks.decimal & has_any_limit & ~censored_any & ~result_null

    # perform configurable data checks
    checks_failed = False

    # check: at least [min_observed_percentage] % of the values are above LOD/LOQ
    if not checks_failed:
        threshold = np.where(has_loq, loq_np, lod_np)
        count_above_lod_loq = np.count_nonzero(
            observed & (biomarker_filled >= threshold)
        )
        denominator = np.count_nonzero(~result_null)
        if (
            denominator == 0
            or count_above_lod_loq
            < denominator / 100.0 * min_observed_percentage
        ):
            checks_failed = True

    # check: at least [min_unique_values] unique values are observed above LOD/LOQ
    if not checks_failed:
        threshold = np.where(has_loq, loq_np, lod_np)
        above_lod_loq = observed & (biomarker_filled >= threshold)
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

        lower[censored.below_lod] = 0
        upper[censored.below_lod] = lod_np[censored.below_lod]

        lower[censored.between_lod_loq] = lod_np[censored.between_lod_loq]
        upper[censored.between_lod_loq] = loq_np[censored.between_lod_loq]

        lower[censored.below_loq] = 0
        upper[censored.below_loq] = loq_np[censored.below_loq]

        fit_values = np.where(censored_any, upper, biomarker_filled)
        fit_mask = observed | censored_any
        dist = fit_censored_lognorm(
            fit_values[fit_mask], censored_any[fit_mask]
        )
        rng = np.random.default_rng(seed=seed)

        cdf_lo = dist.cdf(lower)
        cdf_hi = dist.cdf(upper)

        u = np.zeros_like(biomarker_filled, dtype=float)
        u[censored_any] = rng.uniform(
            cdf_lo[censored_any], cdf_hi[censored_any]
        )
        imputed = np.full_like(biomarker_filled, np.nan, dtype=float)
        imputed[censored_any] = dist.ppf(u[censored_any])

        result = biomarker_filled.copy()
        result[censored_any] = imputed[censored_any]
        result[result_null] = np.nan

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
