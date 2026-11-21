#!/usr/bin/env Rscript
# gen_height_full.R -- full-scale upstream run on the bundled height GWAS,
# reproducing examples/examples.md of the upstream repository.
#
# Usage:
#   Rscript --vanilla gen_height_full.R <tools_dir> <upstream_dir> <work_dir> <models> <threads> \
#           <out_dir_full> <out_dir_summary> <stdout_log> [m3_start=examples|computed]
#
#   models          "2", "3" or "2,3"
#   threads         OpenMP threads handed to upstream as `cores`
#   out_dir_full    receives height_m{2,3}.json (+ per-SNP score binaries)   [large]
#   out_dir_summary receives height_full_m{2,3}.json (no per-SNP arrays)     [small]
#   stdout_log      the file this process' stdout is redirected to; the EM
#                   trace that upstream prints is read back from it.
#
# The genuine upstream genesis() is called (with the shims documented in
# lib/build_upstream.R); its local variables are snapshotted at exit.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 8) stop("usage: gen_height_full.R <tools_dir> <upstream_dir> <work_dir> <models> <threads> <out_dir_full> <out_dir_summary> <stdout_log> [m3_start]")
tools <- args[1]; upstream <- args[2]; work <- args[3]
models <- as.integer(strsplit(args[4], ",")[[1]]); threads <- as.integer(args[5])
out_full <- args[6]; out_sum <- args[7]; log_path <- args[8]
m3_start_mode <- if (length(args) >= 9) args[9] else "examples"
.libPaths(c(file.path(work, "rlib"), .libPaths()))
source(file.path(tools, "lib", "build_upstream.R"))
source(file.path(tools, "lib", "oracle_utils.R"))
dir.create(out_full, recursive = TRUE, showWarnings = FALSE)
dir.create(out_sum, recursive = TRUE, showWarnings = FALSE)

up <- build_upstream(upstream, work, tools, openmp = "auto")
e <- new.env(); load(file.path(upstream, "data", "heightGWAS.RData"), envir = e)
heightGWAS <- e$heightGWAS
# Dry-run aid: GENESIS_ORACLE_SUBSET=<n> keeps only the first n rows so the
# whole script can be exercised in seconds.  Never set it for real fixtures.
subset_n <- suppressWarnings(as.integer(Sys.getenv("GENESIS_ORACLE_SUBSET", "")))
if (!is.na(subset_n)) {
  heightGWAS <- heightGWAS[seq_len(min(subset_n, nrow(heightGWAS))), ]
  .oracle_msg("DRY RUN on the first ", nrow(heightGWAS), " rows of heightGWAS")
}

# Values printed in upstream examples/examples.md ("the est and v should have below values").
ex <- list(
  est2 = c(4.906797e-03, 5.733005e-05, 1.861491e-06),
  v2 = matrix(c(2.870170e-07, -2.426575e-09, -6.077209e-11, -2.426575e-09, 2.491765e-11, 4.715283e-13,
                -6.077209e-11, 4.715283e-13, 1.720162e-14), 3, 3),
  est3 = c(8.899809e-03, 9.476025e-02, 1.458650e-04, 2.227118e-05, 1.567643e-06),
  v3 = matrix(c(1.327856e-06, -1.131049e-05, 6.912489e-10, -2.901301e-09, -9.388865e-11,
                -8.568269e-06, 3.380985e-04, -1.543479e-07, 1.771036e-09, 8.989668e-10,
                -1.846216e-09, -1.353542e-07, 1.517545e-10, 1.166563e-11, 1.901686e-14,
                -3.068113e-09, 9.520417e-09, 5.633801e-12, 9.427492e-12, 1.220433e-13,
                -9.111521e-11, 1.063848e-09, -1.526281e-13, 1.042123e-13, 1.406410e-14), 5, 5))

read_trace_between <- function(log_path, tag) {
  flush(stdout())
  ln <- readLines(log_path, warn = FALSE)
  b <- which(ln == sprintf("### EM-TRACE-BEGIN %s", tag)); en <- which(ln == sprintf("### EM-TRACE-END %s", tag))
  if (!length(b) || !length(en)) return(character())
  ln[(b[length(b)] + 1):(en[length(en)] - 1)]
}

run_model <- function(comp, gen_args, tag) {
  cat(sprintf("### EM-TRACE-BEGIN %s\n", tag)); flush(stdout())
  t0 <- Sys.time()
  cap <- call_with_locals(up, "genesis", gen_args, capture = FALSE)
  wall <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  cat(sprintf("### EM-TRACE-END %s\n", tag)); flush(stdout())
  cap$printed <- read_trace_between(log_path, tag)
  L <- cap$locals
  out <- genesis_locals_to_list(cap, per_snp = "head", n_head = 200, keep_snp_order = FALSE)
  out$wall_seconds_total <- wall
  out$upstream_runhour_em_only <- num(L$runhour)
  out$call <- list(summarydata = "heightGWAS (as bundled; factor columns)", filter = FALSE, modelcomponents = comp,
                   cores = threads, LDcutoff = 0.1, LDwindow = 1, M = 1070777, c0 = 10,
                   print = TRUE, printfreq = 1, qqplot = FALSE,
                   startingpic = if (is.null(gen_args$startingpic)) "NA" else gen_args$startingpic,
                   starting = if (is.null(gen_args$starting)) "NA" else gen_args$starting)
  # deterministic downstream functions at the fitted values
  est <- num(L$est); v <- L$var_est
  out$projection_n253288 <- up$projection(est, v = NULL, n = 253288, CI = FALSE)
  out$numInterval_0.005_Inf <- up$numInterval(0.005, Inf, est)
  pm <- up$postmarginal(L$df, est, c0 = num(L$c0))
  out$postmarginal_first200 <- list(snp = as.character(pm$snp[1:200]), prob_beta0 = num(pm$prob.beta0[1:200]),
                                    posteriormean_marginal = num(pm$posteriormean.marginal[1:200]),
                                    c0_used = num(L$c0))
  out$postmarginal_sums <- list(sum_prob_beta0 = sum(pm$prob.beta0), sum_abs_posteriormean = sum(abs(pm$posteriormean.marginal)),
                                n_nonfinite = sum(!is.finite(pm$prob.beta0)) + sum(!is.finite(pm$posteriormean.marginal)))
  list(out = out, cap = cap)
}

rel <- function(a, b) (a - b) / abs(b)

for (comp in models) {
  tag <- sprintf("m%d", comp)
  if (comp == 2) {
    ga <- list(heightGWAS, filter = FALSE, modelcomponents = 2, cores = threads, LDcutoff = 0.1, LDwindow = 1,
               c0 = 10, startingpic = 0.005, print = TRUE, printfreq = 1, qqplot = FALSE)
    r <- run_model(2, ga, tag)
    out <- r$out; L <- r$cap$locals
    out$examples_md <- list(est = ex$est2, v = mat(ex$v2),
                            rel_diff_est = num(rel(num(L$est), ex$est2)),
                            rel_diff_v = mat(rel(L$var_est, ex$v2)),
                            max_abs_rel_diff_est = max(abs(rel(num(L$est), ex$est2))),
                            max_abs_rel_diff_v = max(abs(rel(L$var_est, ex$v2))),
                            note = "examples.md prints 7 significant digits, so agreement is limited to ~1e-7 relative")
    # the examples.md follow-up calls, evaluated at the fitted est (incl. the example's own use of est[3] as Sig2s)
    est <- num(L$est)
    out$examples_md_calls <- list(
      polyriskpredict_optimum_as_written = prp_json(up$polyriskpredict(N = 253288, Ps = c(0.5, 0.5), Sig2s = c(est[3], est[3]), M = 1070777, M1 = 1070777 * est[1], type = "optimum", k.fold = 3:5)),
      polyriskpredict_GWAS_as_written = prp_json(up$polyriskpredict(N = 253288, Ps = c(0.5, 0.5), Sig2s = c(est[3], est[3]), M = 1070777, M1 = 1070777 * est[1], type = "GWAS", alp.GWAS = 5e-8, k.fold = 3:5)),
      polyriskpredict_optimum_sigmasq = prp_json(up$polyriskpredict(N = 253288, Ps = c(0.5, 0.5), Sig2s = c(est[2], est[2]), M = 1070777, M1 = 1070777 * est[1], type = "optimum", k.fold = 3:5)),
      polyriskpredict_GWAS_sigmasq = prp_json(up$polyriskpredict(N = 253288, Ps = c(0.5, 0.5), Sig2s = c(est[2], est[2]), M = 1070777, M1 = 1070777 * est[1], type = "GWAS", alp.GWAS = 5e-8, k.fold = 3:5)),
      note = "examples.md passes Sig2s=c(est[3],est[3]) for the 2-component model, i.e. the stratification term a rather than sigmasq (est[2]); both variants are stored")
  } else {
    if (m3_start_mode == "computed") {
      prev <- jsonlite::fromJSON(file.path(out_sum, "height_full_m2.json"))
      est2 <- as.numeric(prev$est); src <- "M2 est computed by this oracle (height_full_m2.json)"
    } else {
      est2 <- ex$est2; src <- "M2 est hard-coded in examples.md (7 significant digits), as a user running the example top to bottom would get"
    }
    starting <- rep(0, 5)
    starting[1] <- est2[1]; starting[2] <- 1/9; starting[3] <- est2[2] * 5; starting[4] <- starting[3] / 10; starting[5] <- est2[3]
    ga <- list(heightGWAS, filter = FALSE, modelcomponents = 3, cores = threads, LDcutoff = 0.1, LDwindow = 1,
               c0 = 10, starting = starting, print = TRUE, printfreq = 1, qqplot = FALSE)
    r <- run_model(3, ga, tag)
    out <- r$out; L <- r$cap$locals
    out$starting_source <- src
    out$examples_md <- list(est = ex$est3, v = mat(ex$v3),
                            rel_diff_est = num(rel(num(L$est), ex$est3)),
                            rel_diff_v = mat(rel(L$var_est, ex$v3)),
                            max_abs_rel_diff_est = max(abs(rel(num(L$est), ex$est3))),
                            max_abs_rel_diff_v = max(abs(rel(L$var_est, ex$v3))),
                            note = "examples.md prints 7 significant digits; v is asymmetric in examples.md as well")
    est <- num(L$est)
    out$examples_md_calls <- list(
      polyriskpredict_optimum = prp_json(up$polyriskpredict(N = 253288, Ps = c(est[2], 1 - est[2]), Sig2s = c(est[3], est[4]), M = 1070777, M1 = 1070777 * est[1], type = "optimum", k.fold = 3:5)),
      polyriskpredict_GWAS = prp_json(up$polyriskpredict(N = 253288, Ps = c(est[2], 1 - est[2]), Sig2s = c(est[3], est[4]), M = 1070777, M1 = 1070777 * est[1], type = "GWAS", alp.GWAS = 5e-8, k.fold = 3:5)))
  }
  out$meta <- oracle_meta(up, upstream, threads = threads, extra = list(
    fixture = sprintf("full-scale height GWAS, %d-component model (upstream examples/examples.md)", comp),
    dry_run_subset_rows = if (is.na(subset_n)) "none (full data)" else subset_n,
    determinism_note = "OpenMP reductions over long double with >1 thread: results are reproducible to ~1e-12 relative, not bitwise"))
  # small summary first (so it exists even if the big dump fails); the raw
  # printed lines stay in the full dump only, the summary keeps the parsed matrix
  printed_raw <- out$printed_trace
  out$printed_trace <- NULL
  out$printed_trace_note <- "printed_trace_matrix holds what upstream printed with printfreq=1 (6 significant digits); column 2 is the log-likelihood at the parameters BEFORE that iteration's update"
  write_json(out, file.path(out_sum, sprintf("height_full_m%d.json", comp)))
  cat(sprintf("### SUMMARY m%d: est = %s ; c0 = %g ; last i = %g ; wall = %.1f s\n", comp,
              paste(sprintf("%.10g", num(L$est)), collapse = ", "), num(L$c0), num(L$fit)[1], out$wall_seconds_total))
  cat(sprintf("### SUMMARY m%d: max|rel diff est vs examples.md| = %.3g ; max|rel diff v| = %.3g\n", comp,
              out$examples_md$max_abs_rel_diff_est, out$examples_md$max_abs_rel_diff_v))
  flush(stdout())

  # full dump: adds SNP order and complete per-SNP score matrices
  full <- out
  full$printed_trace <- printed_raw
  full$snp_order <- as.character(L$SNP)
  full$m_S_rows <- mat(L$m_S); full$m_Sbar_rows <- mat(L$m_Sbar); full$per_snp_rows <- "full"
  full$betahat <- num(L$betahat); full$varbetahat <- num(L$varbetahat)
  full$ldscore <- num(L$ldscore); full$Nstar <- num(L$Nstar)
  write_json(full, file.path(out_full, sprintf("height_m%d.json", comp)))
  writeBin(as.vector(t(L$m_S)), file.path(out_full, sprintf("height_m%d_scores_rowmajor.f64", comp)), size = 8, endian = "little")
  writeBin(as.vector(t(L$m_Sbar)), file.path(out_full, sprintf("height_m%d_scorebar_rowmajor.f64", comp)), size = 8, endian = "little")
  rm(full, r); gc()

  # full-precision trace of the first EM iterations (Meps scan; deterministic)
  n_scan <- if (comp == 2) 10L else 4L
  tol <- out$tolerances
  scan <- vector("list", n_scan)
  for (m in seq_len(n_scan)) {
    res <- if (comp == 2)
      quiet(up$EM_func(out$starting, num(L$betahat), num(L$varbetahat), num(L$ldscore), num(L$Nstar), 1070777, 10,
                       tol$pic, tol$sigsq1, tol$a, tol$llk, m, tol$steps, threads, FALSE, 1, TRUE))
    else
      quiet(up$EM_func3(out$starting, c(1e-7, 1e-7), c(0.5, 0.5), num(L$betahat), num(L$varbetahat), num(L$ldscore), num(L$Nstar), 1070777, 10,
                        tol$pic, tol$p1, tol$sigsq1, tol$sigsq2, tol$a, tol$llk, m, tol$steps, threads, FALSE, 1, TRUE))
    scan[[m]] <- list(Meps = m, result = num(res))
    cat(sprintf("### Meps scan m%d: Meps=%d done\n", comp, m)); flush(stdout())
  }
  out$em_trace_first_iterations <- scan
  write_json(out, file.path(out_sum, sprintf("height_full_m%d.json", comp)))
  cat(sprintf("### DONE m%d\n", comp)); flush(stdout())
}
