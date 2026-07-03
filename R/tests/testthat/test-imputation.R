to_logical_with_na <- function(x) {
  vapply(
    x,
    function(v) {
      if (is.null(v)) return(NA)
      as.logical(v)
    },
    logical(1)
  )
}

expect_null_or_between <- function(value, lower, upper) {
  expect_false(is.na(value))
  expect_true(value >= lower)
  expect_true(value <= upper)
}

test_that("medium_bound_imputation follows decision table", {
  skip_if_not_installed("polars")

  measurements <- c(-10, -3, -2, -1, 0.5, 1.5, 3)
  out <- compehndly_apply(
    "medium_bound_imputation",
    measurement = polars::pl$Series(rep(measurements, 4)),
    lod = polars::pl$Series(c(rep(NA_real_, 7), rep(1, 7), rep(NA_real_, 7), rep(1, 7))),
    loq = polars::pl$Series(c(rep(NA_real_, 14), rep(2, 14)))
  )

  expect_equal(
    to_numeric_with_na(out$to_list()),
    c(
      rep(NA_real_, 7),
      NA_real_, NA_real_, NA_real_, 0.5, 0.5, 1.5, 3,
      NA_real_, 1, NA_real_, NA_real_, 1, 1, 3,
      NA_real_, NA_real_, 1.5, 0.5, 0.5, 1.5, 3
    )
  )
})

test_that("lab_sensitivity_dichotomization follows decision table", {
  skip_if_not_installed("polars")

  measurements <- c(-10, -3, -2, -1, 0.5, 1.5, 3)
  out <- compehndly_apply(
    "lab_sensitivity_dichotomization",
    measurement = polars::pl$Series(rep(measurements, 4)),
    lod = polars::pl$Series(c(rep(NA_real_, 7), rep(1, 7), rep(NA_real_, 7), rep(1, 7))),
    loq = polars::pl$Series(c(rep(NA_real_, 14), rep(2, 14)))
  )

  expect_equal(
    to_logical_with_na(out$to_list()),
    c(
      rep(NA, 7),
      NA, NA, NA, FALSE, FALSE, TRUE, TRUE,
      NA, FALSE, NA, NA, FALSE, FALSE, TRUE,
      NA, NA, FALSE, FALSE, FALSE, FALSE, TRUE
    )
  )
})

test_that("random_single_imputation follows null and bound decisions", {
  skip_if_not_installed("polars")

  measurements <- c(-10, -3, -2, -1, 0.5, 1.5, 3)
  out <- compehndly_apply(
    "random_single_imputation",
    biomarker = polars::pl$Series(rep(measurements, 4)),
    lod = polars::pl$Series(c(rep(NA_real_, 7), rep(1, 7), rep(NA_real_, 7), rep(1, 7))),
    loq = polars::pl$Series(c(rep(NA_real_, 14), rep(2, 14))),
    .params = list(seed = 42)
  )
  values <- to_numeric_with_na(out$to_list())

  expect_true(all(is.na(values[1:7])))
  expect_true(all(is.na(values[8:10])))
  expect_null_or_between(values[[11]], 0, 1)
  expect_null_or_between(values[[12]], 0, 1)
  expect_equal(values[13:14], c(1.5, 3))
  expect_true(is.na(values[[15]]))
  expect_null_or_between(values[[16]], 0, 2)
  expect_true(all(is.na(values[17:18])))
  expect_null_or_between(values[[19]], 0, 2)
  expect_null_or_between(values[[20]], 0, 2)
  expect_equal(values[[21]], 3)
  expect_true(all(is.na(values[22:23])))
  expect_null_or_between(values[[24]], 1, 2)
  expect_null_or_between(values[[25]], 0, 1)
  expect_null_or_between(values[[26]], 0, 1)
  expect_null_or_between(values[[27]], 1, 2)
  expect_equal(values[[28]], 3)
})
