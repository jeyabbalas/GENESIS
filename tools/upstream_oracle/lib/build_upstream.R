# build_upstream.R -- compile and load the ORIGINAL upstream GENESIS code
# (https://github.com/yandorazhang/GENESIS) without installing it as a package.
#
# Nothing in this file re-implements upstream numerics.  It
#   1. copies src/EM_EM.cpp out of an upstream clone and applies the recorded
#      build-only patch (patches/EM_EM.cpp.patch),
#   2. compiles the copy with Rcpp::sourceCpp into a private environment,
#   3. sys.source()s the upstream R files into that same environment, and
#   4. installs small shims for the side-effecting calls upstream makes
#      (data(), library(doParallel), makeCluster(), registerDoParallel(),
#      foreach/%do%) so that the genuine upstream functions can be called
#      as-is on arbitrary inputs.
#
# Usage (from another script):
#   source("lib/build_upstream.R")
#   up <- build_upstream(upstream_dir = "/path/to/GENESIS-clone",
#                        work_dir     = "/path/to/scratch/oracle",
#                        tools_dir    = "/path/to/tools/upstream_oracle")
#   up$loglikelihood(...); up$genesis(...)

.oracle_msg <- function(...) cat(sprintf("[oracle] %s\n", paste0(..., collapse = "")), file = stderr())

# Locate an OpenMP header + runtime usable with the current compiler.
# Returns list(mode = "real"|"stub", cxxflags = , libs = , note = ).
detect_openmp <- function(tools_dir, work_dir, openmp = c("auto", "real", "stub")) {
  openmp <- match.arg(openmp)
  stub <- list(mode = "stub",
               cxxflags = sprintf("-I%s -Wno-unknown-pragmas", shQuote(file.path(tools_dir, "stub"))),
               libs = "",
               note = "OpenMP stubbed: omp.h replaced by tools/upstream_oracle/stub/omp.h; all loops serial")
  if (openmp == "stub") return(stub)
  sysname <- Sys.info()[["sysname"]]
  if (sysname == "Darwin") {
    # Apple clang understands OpenMP pragmas with `-Xclang -fopenmp` but ships
    # neither omp.h nor libomp.  CRAN's R for macOS bundles libomp.dylib in
    # $R_HOME/lib; the matching header comes from the tarball published at
    # https://mac.r-project.org/openmp/ (unpacked under <work_dir>/omp).
    inc_candidates <- c(Sys.getenv("GENESIS_ORACLE_OMP_INCLUDE", unset = NA),
                        file.path(work_dir, "omp", "usr", "local", "include"),
                        "/usr/local/include", "/opt/homebrew/opt/libomp/include")
    lib_candidates <- c(Sys.getenv("GENESIS_ORACLE_OMP_LIBDIR", unset = NA),
                        file.path(R.home("lib")),
                        file.path(work_dir, "omp", "usr", "local", "lib"),
                        "/usr/local/lib", "/opt/homebrew/opt/libomp/lib")
    inc <- Filter(function(d) !is.na(d) && file.exists(file.path(d, "omp.h")), inc_candidates)
    lib <- Filter(function(d) !is.na(d) && file.exists(file.path(d, "libomp.dylib")), lib_candidates)
    if (length(inc) && length(lib)) {
      return(list(mode = "real",
                  cxxflags = sprintf("-Xclang -fopenmp -I%s", shQuote(inc[[1]])),
                  libs = sprintf("-L%s -lomp", shQuote(lib[[1]])),
                  note = sprintf("real OpenMP: Apple clang -Xclang -fopenmp, omp.h from %s, libomp.dylib from %s",
                                 inc[[1]], lib[[1]])))
    }
    if (openmp == "real") stop("openmp='real' requested but omp.h/libomp.dylib not found")
    return(stub)
  }
  # Linux / other: the system compiler is assumed to support -fopenmp.
  list(mode = "real", cxxflags = "-fopenmp", libs = "-fopenmp",
       note = "real OpenMP: -fopenmp")
}

build_upstream <- function(upstream_dir, work_dir, tools_dir,
                           openmp = c("auto", "real", "stub"),
                           data_dir = file.path(upstream_dir, "data"),
                           verbose = FALSE) {
  openmp <- match.arg(openmp)
  stopifnot(dir.exists(upstream_dir), dir.exists(tools_dir))
  upstream_dir <- normalizePath(upstream_dir); tools_dir <- normalizePath(tools_dir)
  dir.create(work_dir, recursive = TRUE, showWarnings = FALSE)
  work_dir <- normalizePath(work_dir)
  src_dir <- file.path(work_dir, "src"); dir.create(src_dir, showWarnings = FALSE)

  suppressPackageStartupMessages({
    requireNamespace("Rcpp"); requireNamespace("RcppArmadillo")
  })

  # ---- 1. patched copy of the C++ core --------------------------------------
  orig <- file.path(upstream_dir, "src", "EM_EM.cpp")
  patch_file <- file.path(tools_dir, "patches", "EM_EM.cpp.patch")
  cpp <- file.path(src_dir, "EM_EM.cpp")
  tmp <- tempfile(fileext = ".cpp")
  st <- system2("patch", c("-s", "-o", shQuote(tmp), shQuote(orig), shQuote(patch_file)))
  if (st != 0) stop("recorded patch does not apply cleanly to ", orig)
  # Only rewrite when content changed so that sourceCpp's cache stays valid.
  if (!file.exists(cpp) || !identical(unname(tools::md5sum(tmp)), unname(tools::md5sum(cpp)))) {
    file.copy(tmp, cpp, overwrite = TRUE)
  }
  unlink(tmp)

  # ---- 2. build flags (environment only; no further source changes) ---------
  omp <- detect_openmp(tools_dir, work_dir, openmp)
  .oracle_msg(omp$note)
  # Upstream's cdnorm() uses R's legacy `PI` macro.  R's own header
  # (R_ext/Constants.h) used to provide it as `#define PI M_PI`; that define
  # is commented out in current R (seen with R 4.6.2), so the identical
  # definition is supplied on the command line.  M_PI is a double-precision
  # literal in both cases, so the value seen by the code is unchanged.
  Sys.setenv(PKG_CXXFLAGS = paste("-DPI=M_PI", omp$cxxflags),
             PKG_LIBS = omp$libs)
  # CRAN's macOS R records FLIBS pointing at /opt/gfortran, which need not be
  # installed.  The upstream code needs no Fortran runtime at link time, so
  # blank FLIBS through a private Makevars when that toolchain is absent.
  flibs <- tryCatch(system2(file.path(R.home("bin"), "R"), c("CMD", "config", "FLIBS"),
                            stdout = TRUE, stderr = FALSE), error = function(e) "")
  fdirs <- regmatches(flibs, gregexpr("-L[^ ]+", flibs))[[1]]
  fdirs <- sub("^-L", "", fdirs)
  if (length(fdirs) && !all(dir.exists(fdirs))) {
    mk <- file.path(work_dir, "Makevars.user")
    writeLines("FLIBS=", mk)
    Sys.setenv(R_MAKEVARS_USER = mk)
    .oracle_msg("gfortran toolchain dir(s) missing -> FLIBS blanked via ", mk)
  }

  up <- new.env(parent = globalenv())
  Rcpp::sourceCpp(cpp, env = up, cacheDir = file.path(work_dir, "sourceCpp_cache"),
                  verbose = verbose, showOutput = verbose)

  # ---- 3. upstream R sources -------------------------------------------------
  rfiles <- c("preprocessing.R", "genesis.R", "h2transfer.R", "numInterval.R", "dmixssnp.R",
              "projection.R", "polyriskpredict.R", "postmarginal.R", "qqplotdata.simu.R",
              "qqplot.plot.R", "futuregc.R")
  for (f in rfiles) sys.source(file.path(upstream_dir, "R", f), envir = up, keep.source = TRUE)

  # ---- 4. shims for side effects --------------------------------------------
  up$.oracle <- new.env(parent = emptyenv())
  up$.oracle$data_dir <- data_dir
  up$.oracle$overrides <- list()     # name -> named list of objects (replaces <name>.RData)
  up$.oracle$data_log <- character() # every dataset name requested through data()
  up$.oracle$openmp <- omp
  up$.oracle$cpp <- cpp

  # data(): upstream loads its bundled datasets by name.  The shim loads the
  # corresponding <data_dir>/<name>.RData (or a registered override) into the
  # environment that encloses the upstream functions, which is where the real
  # data() would make them visible to those functions.
  up$data <- function(..., list = character()) {
    nms <- c(as.character(substitute(list(...))[-1L]), list)
    o <- get(".oracle", envir = up)
    for (nm in nms) {
      o$data_log <- c(o$data_log, nm)
      if (!is.null(o$overrides[[nm]])) {
        ov <- o$overrides[[nm]]
        for (v in names(ov)) assign(v, ov[[v]], envir = up)
      } else {
        f <- file.path(o$data_dir, paste0(nm, ".RData"))
        if (!file.exists(f)) stop("data shim: no override and no file for dataset '", nm, "'")
        load(f, envir = up)
      }
    }
    invisible(nms)
  }
  # library(doParallel) / library(foreach): upstream attaches these only to
  # create a PSOCK cluster that is never used for computation and never
  # stopped.  Everything else (library(MASS) in projection()) is passed on.
  up$library <- function(package, ...) {
    pkg <- as.character(substitute(package))
    if (pkg %in% c("doParallel", "foreach")) return(invisible(NULL))
    base::library(pkg, character.only = TRUE, ...)
  }
  up$makeCluster <- function(...) NULL
  up$registerDoParallel <- function(...) invisible(NULL)
  # Sequential foreach(i = ...) %do% {...}: evaluates the body in the calling
  # frame with the loop variable assigned there, exactly like foreach's %do%.
  up$foreach <- function(...) {
    a <- list(...)
    structure(list(var = names(a)[1], values = a[[1]]), class = "oracle_foreach")
  }
  up$`%do%` <- function(obj, ex) {
    ex <- substitute(ex); pf <- parent.frame()
    for (v in obj$values) { assign(obj$var, v, envir = pf); eval(ex, pf) }
    invisible(NULL)
  }
  up
}

# Register an in-memory replacement for one of upstream's bundled datasets,
# e.g. set_data_override(up, "LDwindow1MB_cutoff0.1", list(dataLD = my_table)).
set_data_override <- function(up, name, objects) {
  stopifnot(is.list(objects), !is.null(names(objects)))
  up$.oracle$overrides[[name]] <- objects
  invisible(NULL)
}
clear_data_overrides <- function(up) { up$.oracle$overrides <- list(); invisible(NULL) }

# Call one of the upstream closures and return, in addition to its value,
# a snapshot of every local variable at exit (via base::trace's exit hook, so
# the function body itself is untouched).
call_with_locals <- function(up, fname, args, capture = TRUE) {
  cap <- new.env(parent = emptyenv())
  assign(".oracle_cap", cap, envir = up)
  suppressMessages(trace(fname, where = up, print = FALSE,
                         exit = quote(assign("locals", mget(ls()), envir = .oracle_cap))))
  on.exit({
    suppressMessages(untrace(fname, where = up))
    rm(".oracle_cap", envir = up)
  }, add = TRUE)
  out <- NULL
  if (capture) {
    # everything upstream prints (Rcout / print) is collected line by line
    printed <- utils::capture.output(out <- do.call(get(fname, envir = up), args))
  } else {
    # long runs: let upstream print straight to stdout (live progress)
    printed <- NULL
    out <- do.call(get(fname, envir = up), args)
  }
  list(value = out, locals = cap$locals, printed = printed)
}
