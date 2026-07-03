In each of these tables the column headers indicate the information provided as measurement.

# Imputation Table

In table headers, `both` means both LOD and LOQ.

| Limits available | -10  | -3     | -2     | -1     | value < both | LOD < value < LOQ | value ≥ both |
| ---------------- | ---- | ------ | ------ | ------ | ------------ | ----------------- | ------------ |
| None             | null | null   | null   | null   | null         | null              | null         |
| LOD              | null | null   | null   | Impute | Impute       | null              | Keep value   |
| LOQ              | null | Impute | null   | null   | Impute       | null              | Keep value   |
| LOD and LOQ      | null | null   | Impute | Impute | Impute       | Impute            | Keep value   |

# Dichotomization and Detection Count Table (`_bin` calculation)

| Limits available | -10  | -3    | -2    | -1    | value < both | LOD < value < LOQ | value ≥ both |
| ---------------- | ---- | ----- | ----- | ----- | ------------ | ----------------- | ------------ |
| None             | null | null  | null  | null  | null         | null              | null         |
| LOD              | null | null  | null  | False | False        | null              | True         |
| LOQ              | null | False | null  | null  | False        | null              | True         |
| LOD and LOQ      | null | null  | False | False | False        | False             | True         |

# Code Definitions

| Code | Meaning             |
| ---- | ------------------- |
| -10  | Not measured        |
| -1   | Below LOD           |
| -2   | Between LOD and LOQ |
| -3   | Below LOQ           |

`-1` is used when LOD is known and LOQ is known or unknown. `-2` is used when both
LOD and LOQ are known. `-3` is used when LOD is unknown and LOQ is known.

For `-10`, the measurement was planned but unexpected reasons prevented the sample
from being measured (e.g. insufficient sample, broken tube, analytical issue).

**Note:** Decimal values may occur in the LOD/LOQ range.
