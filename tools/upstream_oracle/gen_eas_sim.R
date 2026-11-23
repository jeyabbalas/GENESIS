#!/usr/bin/env Rscript
# gen_eas_sim.R -- full-scale run of the ORIGINAL GENESISasian package
# (https://github.com/yandorazhang/GENESISasian, commit 4df90ac6ef3ea84219110187ef4ef5b5a79b83e6)
# with its own defaults (M = 953834, LD 1 Mb / r2 >= 0.1, LDSC starting values) on the
# simulated East Asian GWAS written by make_eas_sim.py.
#
# Usage:
#   Rscript --vanilla gen_eas_sim.R <tools_dir> <genesisasian_dir> <work_dir> <sumstats.tsv.gz> \
#           <models> <threads> <out_dir_summary> <stdout_log>
#
#   models          "2", "3" or "2,3"
#   out_dir_summary receives eas_sim_m{2,3}.json
#   stdout_log      the file this process' stdout is redirected to (EM trace is read back)
#
# GENESISasian's src/EM_EM.cpp is byte-identical to GENESIS's, so the same build recipe
# (lib/build_upstream.R, patches/EM_EM.cpp.patch) applies; its R/genesis.R differs only
# in the default M. The genuine genesis() is called and its locals snapshotted at exit.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 8) stop("usage: gen_eas_sim.R <tools_dir> <genesisasian_dir> <work_dir> <sumstats> <models> <threads> <out_dir_summary> <stdout_log>")
tools <- args[1]; upstream <- args[2]; work <- args[3]; sumstats <- args[4]
models <- as.integer(strsplit(args[5], ",")[[1]]); threads <- as.integer(args[6])
out_sum <- args[7]; log_path <- args[8]
.libPaths(c(file.path(work, "rlib"), .libPaths()))
source(file.path(tools, "lib", "build_upstream.R"))
source(file.path(tools, "lib", "oracle_utils.R"))
dir.create(out_sum, recursive = TRUE, showWarnings = FALSE)

up <- build_upstream(upstream, work, tools, openmp = "auto")
stopifnot(formals(up$genesis)$M == 953834)
gwas <- read.table(gzfile(sumstats), header = TRUE, sep = "\t", stringsAsFactors = FALSE,
                   colClasses = c("character", "numeric", "numeric"))
stopifnot(identical(names(gwas), c("snp", "z", "n")))
sumstats_md5 <- unname(tools::md5sum(sumstats))

read_trace_between <- function(log_path, tag) {
  flush(stdout())
  ln <- readLines(log_path, warn = FALSE)
  b <- which(ln == sprintf("### EM-TRACE-BEGIN %s", tag)); en <- which(ln == sprintf("### EM-TRACE-END %s", tag))
  if (!length(b) || !length(en)) return(character())
  ln[(b[length(b)] + 1):(en[length(en)] - 1)]
}

for (comp in models) {
  tag <- sprintf("m%d", comp)
  # everything else is left at GENESISasian's defaults (startingpic = NA, starting = NA,
  # M = 953834, c0 = 10, tolerances NA)
  ga <- list(gwas, filter = FALSE, modelcomponents = comp, cores = threads,
             print = TRUE, printfreq = 1, qqplot = FALSE)
  cat(sprintf("### EM-TRACE-BEGIN %s\n", tag)); flush(stdout())
  t0 <- Sys.time()
  cap <- call_with_locals(up, "genesis", ga, capture = FALSE)
  wall <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  cat(sprintf("### EM-TRACE-END %s\n", tag)); flush(stdout())
  cap$printed <- read_trace_between(log_path, tag)
  L <- cap$locals
  out <- genesis_locals_to_list(cap, per_snp = "head", n_head = 50, keep_snp_order = FALSE)
  out$printed_trace <- NULL
  out$wall_seconds_total <- wall
  out$upstream_runhour_em_only <- num(L$runhour)
  out$M_used <- num(L$M)
  out$sumstats <- list(file = basename(sumstats), md5_gz = sumstats_md5, rows = nrow(gwas),
                       generator = "tools/upstream_oracle/make_eas_sim.py")
  out$call <- list(summarydata = "simulated East Asian GWAS (snp, z, n)", filter = FALSE,
                   modelcomponents = comp, cores = threads, print = TRUE, printfreq = 1, qqplot = FALSE,
                   everything_else = "GENESISasian defaults")
  est <- num(L$est)
  out$projection_n300000 <- up$projection(est, v = NULL, n = 300000, CI = FALSE)
  out$numInterval_0.005_Inf <- up$numInterval(0.005, Inf, est)
  out$meta <- oracle_meta(up, upstream, threads = threads, extra = list(
    fixture = sprintf("GENESISasian, simulated East Asian GWAS, %d-component model", comp),
    determinism_note = "OpenMP reductions over long double with >1 thread: results are reproducible to ~1e-12 relative, not bitwise"))
  out$meta$upstream_repo <- "https://github.com/yandorazhang/GENESISasian"
  write_json(out, file.path(out_sum, sprintf("eas_sim_m%d.json", comp)))
  cat(sprintf("### SUMMARY m%d: est = %s ; c0 = %g ; last i = %g ; M = %g ; wall = %.1f s\n", comp,
              paste(sprintf("%.10g", est), collapse = ", "), num(L$c0), num(L$fit)[1], num(L$M), wall))
  flush(stdout())
}
