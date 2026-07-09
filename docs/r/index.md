# R Overview

The R implementation is intended to mirror the Python derived-variable behavior
and consume the same shared conformance vectors.

Use the R package when you want the same derived-variable functions from an R
analysis workflow. The current R API is Polars-backed and exposes a single
entrypoint:

```r
compehndly_apply(function_name, ..., .params = list())
```

The `function_name` selects the derived-variable function. Positional or named
Polars series supply the inputs, and `.params` supplies scalar function
parameters.

Start with:

- [Usage](usage.md): install dependencies, load the local package, run examples,
  and execute the R tests.
- [Functions](functions.md): behavior notes for the R functions and parity with
  Python.
- [Integration Patterns](integration.md): R workflow patterns as the package
  interface expands.
