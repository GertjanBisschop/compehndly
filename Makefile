.PHONY: python-test r-setup r-setup-core r-setup-polars r-test

R_LIBS_USER ?= $(CURDIR)/.r-lib

python-test:
	cd python && uv run pytest -q

r-setup: r-setup-core r-setup-polars

r-setup-core:
	mkdir -p "$(R_LIBS_USER)"
	R_LIBS_USER="$(R_LIBS_USER)" Rscript -e 'Sys.setenv(USE_BUNDLED_LIBUV="1"); pkgs <- c("pkgload","testthat","jsonlite"); install.packages(pkgs, repos="https://cloud.r-project.org"); missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]; if (length(missing)) stop("Missing R packages after install: ", paste(missing, collapse = ", "))'

r-setup-polars:
	mkdir -p "$(R_LIBS_USER)"
	R_LIBS_USER="$(R_LIBS_USER)" Rscript -e 'Sys.setenv(NOT_CRAN="true"); install.packages("polars", repos=c("https://community.r-multiverse.org","https://rpolars.r-universe.dev","https://cloud.r-project.org")); if (!requireNamespace("polars", quietly = TRUE)) stop("Missing R package after install: polars")'

r-test:
	cd R && R_LIBS_USER="$(R_LIBS_USER)" Rscript tests/testthat.R
