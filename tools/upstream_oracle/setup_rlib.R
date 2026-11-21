#!/usr/bin/env Rscript
# setup_rlib.R -- install the R packages the oracle needs into an ISOLATED
# library (never the user's or the system library).
#
# Usage: Rscript --vanilla setup_rlib.R <work_dir>
#   -> packages go to <work_dir>/rlib
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) stop("usage: setup_rlib.R <work_dir>")
lib <- file.path(args[1], "rlib")
dir.create(lib, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(lib, .libPaths()))
need <- c("Rcpp", "RcppArmadillo", "jsonlite")
have <- rownames(installed.packages(lib.loc = lib))
todo <- setdiff(need, have)
if (length(todo)) {
  # binaries where CRAN provides them (macOS/Windows); source elsewhere
  type <- if (.Platform$pkgType == "source") "source" else "binary"
  install.packages(todo, lib = lib, repos = "https://cloud.r-project.org", type = type)
}
for (p in need) cat(sprintf("%-14s %s\n", p, as.character(packageVersion(p, lib.loc = lib))))
