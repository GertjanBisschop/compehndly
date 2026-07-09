test_that("bin_decoding decodes numbered pairs", {
  skip_if_not_installed("polars")

  out <- compehndly_apply(
    "bin_decoding",
    values = test_series(c(-10, 1.25, -3, 4.5, -2)),
    copy_from_1 = test_series(c(10, 20, 30, 40, 50)),
    copy_from_2 = test_series(c(60, 70, 80, 90, 100)),
    .params = list(filter_value_1 = -10, filter_value_2 = -3)
  )

  expect_equal(
    to_numeric_with_na(series_values(out)),
    c(10, 1.25, 80, 4.5, -2)
  )
})

test_that("bin_decoding requires complete contiguous unique pairs", {
  skip_if_not_installed("polars")

  expect_error(
    compehndly_apply(
      "bin_decoding",
      values = test_series(c(-10)),
      .params = list(filter_value_1 = -10)
    ),
    "missing copy_from_1"
  )

  expect_error(
    compehndly_apply(
      "bin_decoding",
      values = test_series(c(-10)),
      copy_from_2 = test_series(c(10)),
      .params = list(filter_value_2 = -10)
    ),
    "contiguous"
  )

  expect_error(
    compehndly_apply(
      "bin_decoding",
      values = test_series(c(-10)),
      copy_from_1 = test_series(c(10)),
      copy_from_2 = test_series(c(20)),
      .params = list(filter_value_1 = -10, filter_value_2 = -10)
    ),
    "unique"
  )
})
