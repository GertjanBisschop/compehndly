# compehndly (R)

This folder contains the R implementation of derived-variable functions, aligned
with the Python behavior through shared conformance vectors.

## Prerequisites

Install core R tooling:

```r
install.packages(c("pkgload", "testthat", "jsonlite"))
```

Install `polars` for R (required by `compehndly_apply`) from R-multiverse:

```r
Sys.setenv(NOT_CRAN = "true")
install.packages(
  "polars",
  repos = c("https://community.r-multiverse.org", "https://rpolars.r-universe.dev", "https://cloud.r-project.org")
)
```

On Ubuntu/Debian, if compiling dependencies fails, install build tools first:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake libcurl4-openssl-dev libssl-dev libuv1-dev libxml2-dev
```

## Load The Local Package

From the repository root:

```r
pkgload::load_all("R", export_all = TRUE)
```

This loads the functions defined in:

- `R/R/derived_variables.R`

## Run Derived Variables

Main entrypoint:

- `compehndly_apply(function_name, ..., .params = list())`

Example:

```r
library(polars)

measured <- pl$Series("measured", c(50.0, 100.0, 75.0))
sg <- pl$Series("sg_measured", c(1.020, 1.015, 1.025))

out <- compehndly_apply(
  "normalize_specific_gravity",
  measured = measured,
  sg_measured = sg,
  .params = list(sg_ref = 1.024)
)

as.vector(out)
```

Bin decoding uses the same numbered pair contract as Python:

```r
values <- pl$Series("values", c(-10.0, 1.25, -3.0, 4.5))
copy_a <- pl$Series("copy_from_1", c(10.0, 20.0, 30.0, 40.0))
copy_b <- pl$Series("copy_from_2", c(50.0, 60.0, 70.0, 80.0))

out <- compehndly_apply(
  "bin_decoding",
  values = values,
  copy_from_1 = copy_a,
  copy_from_2 = copy_b,
  .params = list(filter_value_1 = -10.0, filter_value_2 = -3.0)
)

as.vector(out)
```

## Run Shared Conformance Tests

Shared vectors live in:

- `shared/conformance/derived_variables_cases.json`

Run R tests from repository root:

```r
testthat::test_dir("R/tests/testthat")
```

Conformance scaffold:

- `R/tests/testthat/test-conformance.R`
