#!/usr/bin/env Rscript
# gen_synthetic.R -- function-level and EM-trace fixtures from the genuine
# upstream code on synthetic inputs.
#
# Usage:
#   Rscript --vanilla gen_synthetic.R <tools_dir> <upstream_dir> <work_dir> <out_dir> [datasets=small,medium]
#
# Writes <out_dir>/synthetic_small.json.gz and <out_dir>/synthetic_medium.json.gz.
# The synthetic INPUTS are stored in the fixtures; consumers must not try to
# regenerate them from the seeds (R's RNG streams are not portable).

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4) stop("usage: gen_synthetic.R <tools_dir> <upstream_dir> <work_dir> <out_dir> [datasets]")
tools <- args[1]; upstream <- args[2]; work <- args[3]; out_dir <- args[4]
which_sets <- if (length(args) >= 5) strsplit(args[5], ",")[[1]] else c("small", "medium")
.libPaths(c(file.path(work, "rlib"), .libPaths()))
source(file.path(tools, "lib", "build_upstream.R"))
source(file.path(tools, "lib", "oracle_utils.R"))
up <- build_upstream(upstream, work, tools, openmp = "auto")
NT <- 1L   # single thread everywhere: bitwise-deterministic reductions

# default tolerances of upstream genesis()
TOL <- list(pic = 1e-6, p1 = 1e-5, sigsq1 = 1e-8, sigsq2 = 1e-8, a = 1e-9, llk = 1e-6, maxEM = 3000, steps = 2)
LOWER_PI <- c(1e-7, 1e-7); UPPER_PI <- c(0.5, 0.5)
N_SCAN <- 60L   # Meps = 1..N_SCAN (3-component model on the medium data set: 40)

# ----------------------------------------------------------------------------
# synthetic data
# ----------------------------------------------------------------------------
make_dataset <- function(name, K, K_drop, M, seed, W = 15L, p_edge = 0.3) {
  set.seed(seed)
  K_ref <- K + K_drop
  truth <- c(pic = 0.01, sigsq = 5e-5, a = 1e-6)
  forced <- c(1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20, 181, 313, 700, 710)
  Nstar <- pmin(710, pmax(1, round(exp(rnorm(K_ref, log(37), 0.72)))))
  forced_pos <- sample.int(K_ref - 5L, length(forced)); Nstar[forced_pos] <- forced
  ld <- pmax(1, Nstar * runif(K_ref, 0.15, 0.7)); ld[Nstar == 1] <- 1
  N <- round(runif(K_ref, 50000, 150000))
  j <- rbinom(K_ref, size = Nstar, prob = truth[["pic"]])
  bh0 <- rnorm(K_ref, mean = 0, sd = sqrt(j * truth[["sigsq"]] * ld / Nstar + 1 / N + truth[["a"]]))
  Z <- bh0 * sqrt(N)
  # extremes (kept in the GWAS): z^2 ~ 79.9 (both signs), z == 0 exactly, z^2 == 25
  ext_pos <- sample(setdiff(seq_len(K_ref - 5L), forced_pos), 6)
  Z[ext_pos] <- c(sqrt(79.9), -sqrt(79.9), 0, 0, 5, -5)
  # one extreme on a forced low-Nstar SNP too
  Z[forced_pos[1]] <- -sqrt(79.9)
  # symmetric neighbourhoods (self + random nearby reference SNPs)
  nb <- lapply(seq_len(K_ref), function(i) i)
  for (i in seq_len(K_ref - 1L)) {
    js <- (i + 1L):min(K_ref, i + W)
    js <- js[runif(length(js)) < p_edge]
    for (jj in js) { nb[[i]] <- c(nb[[i]], jj); nb[[jj]] <- c(nb[[jj]], i) }
  }
  # upstream string format: comma-joined rsIDs, the SNP itself last
  tag <- vapply(seq_len(K_ref), function(i) {
    o <- setdiff(nb[[i]], i); if (length(o) > 1) o <- sample(o)
    paste0("rs", c(o, i), collapse = ",")
  }, "")
  snp <- paste0("rs", seq_len(K_ref))
  dataLD <- data.frame(SNPname = snp, Nstar = as.integer(Nstar), TaggingSNPs = tag,
                       LD.score.correct = ld, CHR = 1L, BP = 1000L * seq_len(K_ref), stringsAsFactors = FALSE)
  # GWAS = reference minus K_drop SNPs: the 5 largest rs numbers (their names in
  # other SNPs' tagging lists then exceed max(rs number) -> NA path of upstream's
  # dictionary) and random others (-> 0 path).
  protected <- c(forced_pos, ext_pos)
  drop <- c((K_ref - 4L):K_ref, sample(setdiff(seq_len(K_ref - 5L), protected), K_drop - 5L))
  stopifnot(length(unique(drop)) == K_drop, !any(drop %in% protected))
  keep <- setdiff(seq_len(K_ref), drop)
  keep <- sample(keep)                      # GWAS rows deliberately NOT in reference order
  sumstats <- data.frame(snp = snp[keep], z = Z[keep], n = N[keep], stringsAsFactors = FALSE)
  list(name = name, seed = seed, K = length(keep), K_ref = K_ref, M = M, truth = truth, dataLD = dataLD,
       sumstats = sumstats, nb = nb, keep = keep, forced_pos = forced_pos, ext_pos = ext_pos)
}

# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
head_rows <- function(m, n) mat(m[seq_len(min(n, nrow(m))), , drop = FALSE])

per_snp_llk <- function(fun, par, d, c0) vapply(seq_len(d$K), function(k)
  num(fun(par, d$betahat[k], d$varbetahat[k], d$ldscore[k], c0, d$Nstar[k], NT)), 0.0)

case_m2 <- function(par, c0, d, size, weight_store, marginal_store, per_snp) {
  ll <- num(up$loglikelihood(par, d$betahat, d$varbetahat, d$ldscore, c0, d$Nstar, NT))
  w <- up$weight(par, d$betahat, d$varbetahat, d$ldscore, c0, d$Nstar, NT)
  wl <- up$weight_loglikelihood(par, d$betahat, d$varbetahat, d$ldscore, c0, d$Nstar, NT)
  s1 <- num(up$onestep_varcomponent(c(par[2], par[3]), w, d$betahat, d$varbetahat, d$ldscore, d$Nstar, NT))
  s2 <- num(up$onestep_varcomponent(s1, w, d$betahat, d$varbetahat, d$ldscore, d$Nstar, NT))
  SS <- up$SS(par, d$betahat, d$varbetahat, d$ldscore, c0, d$Nstar, NT)
  jj <- 0:c0
  out <- list(
    par = par, c0 = c0,
    loglikelihood = ll,
    weight_loglikelihood_llk = num(wl$llk),
    weight_identical_to_weight_loglikelihood_w = identical(w, wl$w),
    weight_dim = dim(w),
    weight_rowsum_minus1_maxabs = max(abs(rowSums(w) - 1)),
    weight_n_nonfinite = sum(!is.finite(w)),
    update_pi1 = num(up$update_pi1(w, d$Nstar, NT)),
    onestep_varcomponent = s1,
    onestep_varcomponent_2nd = s2,
    onestep_note = "onestep_varcomponent(c(sigsq,a), w, ...) with w = weight(par); _2nd applies it again starting from the first result with the SAME w",
    S = num(up$S(par, d$betahat, d$varbetahat, d$ldscore, c0, d$Nstar, NT)),
    SS_colsums = num(colSums(SS)),
    I = mat(up$I(par, d$betahat, d$varbetahat, d$ldscore, c0, d$Nstar, NT)),
    Ik_first5 = lapply(1:5, function(k) mat(up$Ik(par, d$betahat[k], d$varbetahat[k], d$ldscore[k], c0, d$Nstar[k])))
  )
  if (size == "small") { out$SS <- mat(SS); out$SS_rows <- "full" } else { out$SS <- head_rows(SS, 50); out$SS_rows <- "first50" }
  if (per_snp) {
    out$loglikelihood_per_snp <- per_snp_llk(up$loglikelihood, par, d, c0)
    out$weight_Ej <- num(w %*% jj)
  }
  if (weight_store == "full") { out$weight <- mat(w); out$weight_rows <- "full" }
  else if (weight_store != "none") { n <- as.integer(sub("first", "", weight_store)); out$weight <- head_rows(w, n); out$weight_rows <- weight_store }
  if (marginal_store != "none") {
    mm <- up$mixture_components_marginal(par, d$ldscore, c0, d$Nstar, NT)
    if (marginal_store == "full") out$marginal <- list(rows = "full", proportions = mat(mm$proportions), varcomponents = mat(mm$varcomponents))
    else { n <- as.integer(sub("first", "", marginal_store)); out$marginal <- list(rows = marginal_store, proportions = head_rows(mm$proportions, n), varcomponents = head_rows(mm$varcomponents, n)) }
  }
  out
}

case_m3 <- function(par, c0, d, size, weight_store, marginal_store, per_snp) {
  ll <- num(up$loglikelihood3(par, d$betahat, d$varbetahat, d$ldscore, c0, d$Nstar, NT))
  w <- up$weight3(par, d$betahat, d$varbetahat, d$ldscore, c0, d$Nstar, NT)
  wl <- up$weight_loglikelihood3(par, d$betahat, d$varbetahat, d$ldscore, c0, d$Nstar, NT)
  s1 <- num(up$onestep_varcomponent3(c(par[3], par[4], par[5]), w, d$betahat, d$varbetahat, d$ldscore, d$Nstar, NT))
  s2 <- num(up$onestep_varcomponent3(s1, w, d$betahat, d$varbetahat, d$ldscore, d$Nstar, NT))
  SS <- up$SS3(par, d$betahat, d$varbetahat, d$ldscore, c0, d$Nstar, NT)
  j1 <- rep(0:c0, each = c0 + 1); j2 <- rep(0:c0, times = c0 + 1)   # column index = j1*(c0+1)+j2
  out <- list(
    par = par, c0 = c0,
    loglikelihood3 = ll,
    weight_loglikelihood3_llk = num(wl$llk),
    weight_identical_to_weight_loglikelihood_w = identical(w, wl$w),
    weight_dim = dim(w),
    weight_column_layout = "column (0-based) = j1*(c0+1) + j2; entries with j1+j2 > Nstar are 0",
    weight_rowsum_minus1_maxabs = max(abs(rowSums(w) - 1)),
    weight_n_nonfinite = sum(!is.finite(w)),
    update_p3 = num(up$update_p3(w, d$Nstar, NT)),
    onestep_varcomponent3 = s1,
    onestep_varcomponent3_2nd = s2,
    onestep_note = "onestep_varcomponent3(c(sig1,sig2,a), w, ...) with w = weight3(par); _2nd applies it again starting from the first result with the SAME w",
    S3 = num(up$S3(par, d$betahat, d$varbetahat, d$ldscore, c0, d$Nstar)),
    S3_note = "upstream S3 divides every per-SNP score by K before summing (unlike S for the 2-component model)",
    SS3_colsums = num(colSums(SS)),
    I3 = mat(up$I3(par, d$betahat, d$varbetahat, d$ldscore, c0, d$Nstar, NT)),
    I3_note = "exactly as upstream computes it: NOT symmetric (entry [3,1] in R indexing lacks the E_w[r3*r1] term because of `rr(2,0) = rr(2,0);` in Ik3)",
    Ik3_first5 = lapply(1:5, function(k) mat(up$Ik3(par, d$betahat[k], d$varbetahat[k], d$ldscore[k], c0, d$Nstar[k])))
  )
  if (size == "small") { out$SS3 <- mat(SS); out$SS3_rows <- "full" } else { out$SS3 <- head_rows(SS, 50); out$SS3_rows <- "first50" }
  if (per_snp) {
    out$loglikelihood3_per_snp <- per_snp_llk(up$loglikelihood3, par, d, c0)
    out$weight_Ej1 <- num(w %*% j1); out$weight_Ej2 <- num(w %*% j2)
  }
  if (weight_store == "full") { out$weight <- mat(w); out$weight_rows <- "full" }
  else if (weight_store != "none") { n <- as.integer(sub("first", "", weight_store)); out$weight <- head_rows(w, n); out$weight_rows <- weight_store }
  if (marginal_store != "none") {
    mm <- up$mixture_3components_marginal(par, d$ldscore, c0, d$Nstar, NT)
    if (marginal_store == "full") out$marginal <- list(rows = "full", proportions = mat(mm$proportions), varcomponents = mat(mm$varcomponents))
    else { n <- as.integer(sub("first", "", marginal_store)); out$marginal <- list(rows = marginal_store, proportions = head_rows(mm$proportions, n), varcomponents = head_rows(mm$varcomponents, n)) }
  }
  out
}

em_call_m2 <- function(start, c0, strat, d, Meps, tol = TOL)
  num(quiet(up$EM_func(start, d$betahat, d$varbetahat, d$ldscore, d$Nstar, d$M, c0,
                       tol$pic, tol$sigsq1, tol$a, tol$llk, Meps, tol$steps, NT, FALSE, 1, strat)))
em_call_m3 <- function(start, c0, strat, d, Meps, tol = TOL)
  num(quiet(up$EM_func3(start, LOWER_PI, UPPER_PI, d$betahat, d$varbetahat, d$ldscore, d$Nstar, d$M, c0,
                        tol$pic, tol$p1, tol$sigsq1, tol$sigsq2, tol$a, tol$llk, Meps, tol$steps, NT, FALSE, 1, strat)))

# One EM scenario: optional genuine genesis() run (post-processing included) + Meps scan.
em_scenario <- function(comp, label, d, sumstats, size, c0 = 10, strat = TRUE, startingpic = NA, starting = NA,
                        run_genesis = TRUE) {
  .oracle_msg(sprintf("  [%s] M%d scenario '%s'", d$name, comp, label))
  sc <- list(label = label, modelcomponents = comp, c0_start = c0, stratification = strat,
             startingpic = if (is.na(startingpic)) "NA" else startingpic,
             starting_argument = if (all(is.na(starting))) "NA" else starting,
             tolerances = TOL, M = d$M, num_threads = NT,
             lower_pi = if (comp == 3) LOWER_PI else NULL, upper_pi = if (comp == 3) UPPER_PI else NULL)
  start <- starting
  if (run_genesis) {
    ga <- list(sumstats, filter = FALSE, modelcomponents = comp, cores = NT, LDcutoff = 0.1, LDwindow = 1,
               M = d$M, c0 = c0, BIC.gamma = 0.5, print = TRUE, printfreq = 1, starting = starting,
               startingpic = startingpic, tolerance = NA, qqplot = FALSE, stratification = strat)
    cap <- tryCatch(call_with_locals(up, "genesis", ga), error = function(e) e)
    if (inherits(cap, "error")) {
      sc$genesis <- list(status = "error", error_message = conditionMessage(cap),
                         note = "the genuine upstream genesis() call stopped with this R error")
    } else {
      full_scores <- size == "small" && grepl("^(a_|f_|e1_)", label)
      g <- genesis_locals_to_list(cap, per_snp = if (full_scores) "full" else "head", n_head = 50, keep_snp_order = FALSE)
      g$status <- "ok"
      # keep only a compact digest of what upstream printed (the full-precision
      # Meps scan below supersedes the 6-digit printed values)
      ptm <- g$printed_trace_matrix; g$printed_trace <- NULL; g$printed_trace_matrix <- NULL
      if (!is.null(ptm)) {
        ic0 <- if (comp == 2) 7 else 9
        chg <- which(c(TRUE, diff(ptm[, ic0]) != 0))
        g$printed_n_iterations <- nrow(ptm)
        g$printed_c0_changes <- list(iteration0 = I(ptm[chg, 1]), c0 = I(ptm[chg, ic0]),
                                     note = "printed iteration index (0-based) at which the printed c0 first takes each value")
        g$printed_last_line <- ptm[nrow(ptm), ]
      }
      sc$genesis <- g
      start <- num(cap$locals$starting)
    }
  }
  if (all(is.na(start))) {
    # genesis() failed before/without exposing its starting values: recompute them as genesis.R does
    st2 <- ldsc_start_m2(up, d$betahat, d$varbetahat, d$ldscore, d$Nstar, d$M, c0, startingpic, NT)$starting
    if (comp == 2) start <- st2 else {
      e2 <- num(quiet(up$EM_func(st2, d$betahat, d$varbetahat, d$ldscore, d$Nstar, d$M, c0, max(TOL$pic, 1e-6), max(TOL$sigsq1, 1e-7),
                                 max(TOL$a, 1e-7), max(TOL$llk, 1e-5), TOL$maxEM, TOL$steps, NT, FALSE, 1, strat)))[3:5]
      start <- c(e2[1], 1/9, e2[2] * 5, e2[2] * 5 / 10, e2[3])
    }
  }
  sc$em_start <- start
  if (comp == 3 && all(is.na(starting))) {
    # the preliminary 2-component fit genesis() runs to build the M3 start (R/genesis.R lines 141-164)
    st2 <- ldsc_start_m2(up, d$betahat, d$varbetahat, d$ldscore, d$Nstar, d$M, c0, startingpic, NT)
    r2 <- num(quiet(up$EM_func(st2$starting, d$betahat, d$varbetahat, d$ldscore, d$Nstar, d$M, c0, max(TOL$pic, 1e-6),
                               max(TOL$sigsq1, 1e-7), max(TOL$a, 1e-7), max(TOL$llk, 1e-5), TOL$maxEM, TOL$steps, NT, FALSE, 1, strat)))
    e2 <- r2[3:5]
    rebuilt <- c(e2[1], 1.0/9, e2[2] * 5, (e2[2] * 5) / 10, e2[3])
    sc$prelim_m2 <- list(start = st2$starting, ldsc_intercept = st2$ldsc_intercept, ldsc_slope = st2$ldsc_slope,
                         tolerances = list(pic = max(TOL$pic, 1e-6), sigsq1 = max(TOL$sigsq1, 1e-7), a = max(TOL$a, 1e-7),
                                           llk = max(TOL$llk, 1e-5), maxEM = TOL$maxEM, steps = TOL$steps),
                         result = r2, m3_start_rebuilt = rebuilt,
                         rebuilt_identical_to_genesis_starting = identical(rebuilt, start))
  }
  if (comp == 2 && all(is.na(starting))) {
    st2 <- ldsc_start_m2(up, d$betahat, d$varbetahat, d$ldscore, d$Nstar, d$M, c0, startingpic, NT)
    sc$ldsc_start <- list(starting = st2$starting, ldsc_intercept = st2$ldsc_intercept, ldsc_slope = st2$ldsc_slope,
                          llk_at_start = num(st2$llk_at_start), identical_to_genesis_starting = identical(st2$starting, start))
  }
  f <- if (comp == 2) em_call_m2 else em_call_m3
  n_scan <- if (comp == 3 && size != "small") 40L else N_SCAN
  sc$trace <- lapply(seq_len(n_scan), function(m) list(Meps = m, result = f(start, c0, strat, d, m)))
  sc$converged <- list(Meps = TOL$maxEM, result = f(start, c0, strat, d, TOL$maxEM))
  sc$result_layout <- if (comp == 2)
    "i_last (0-based), llk at returned parameters (final c0), pic, sigmasq, a, heritability, c0, 3 CPU timings (ignore)"
    else "i_last (0-based), llk at returned parameters (final c0), pic, p1, sigmasq1, sigmasq2, a, heritability, c0, 3 CPU timings (ignore)"
  if (run_genesis && !is.null(sc$genesis$fit)) {
    nt <- if (comp == 2) 7 else 9   # drop the timing entries before comparing
    sc$converged_identical_to_genesis_fit <- identical(sc$converged$result[1:nt], sc$genesis$fit[1:nt])
  }
  ic <- if (comp == 2) 7 else 9
  c0s <- vapply(sc$trace, function(t) t$result[ic], 0.0)
  sc$c0_path_first_iterations <- c0s
  sc$c0_changed <- any(c0s != c0) || sc$converged$result[ic] != c0
  sc
}

# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
specs <- list(small = list(K = 300L, K_drop = 40L, M = 20000, seed = 20181),
              medium = list(K = 3000L, K_drop = 200L, M = 200000, seed = 20182))
pars2 <- list(c(0.01, 5e-5, 1e-6), c(0.002, 2e-4, 0), c(0.05, 1e-5, -2e-6), c(0.3, 1e-6, 5e-6))
c0s2 <- c(10L, 3L, 20L)
pars3 <- list(c(0.01, 1/9, 2.5e-4, 2.5e-5, 1e-6), c(0.02, 0.3, 1e-4, 1e-5, 0), c(0.005, 0.05, 1e-3, 5e-5, 2e-6))
c0s3 <- c(10L, 3L, 5L)

for (size in which_sets) {
  sp <- specs[[size]]
  .oracle_msg("dataset ", size)
  ds <- make_dataset(size, sp$K, sp$K_drop, sp$M, sp$seed)
  clear_data_overrides(up)
  set_data_override(up, "LDwindow1MB_cutoff0.1", list(dataLD = ds$dataLD))
  df <- up$preprocessing(ds$sumstats, 0.1, 1, FALSE)       # genuine upstream merge
  stopifnot(nrow(df) == ds$K, identical(df$SNP, ds$sumstats$snp))
  d <- list(name = size, K = nrow(df), M = sp$M,
            betahat = as.numeric(as.character(df$betahat)), varbetahat = as.numeric(as.character(df$varbetahat)),
            ldscore = as.numeric(as.character(df$LD.score.correct)), Nstar = as.numeric(as.character(df$Nstar)))
  stopifnot(-2e-6 > -min(d$varbetahat))
  # effective neighbourhoods among GWAS rows (0-based df-row indices, self included, ascending)
  ref_idx <- as.integer(sub("^rs", "", df$SNP))
  row_of_ref <- rep(NA_integer_, ds$K_ref); row_of_ref[ref_idx] <- seq_len(d$K)
  tag_rows <- lapply(seq_len(d$K), function(k) { r <- row_of_ref[ds$nb[[ref_idx[k]]]]; sort(r[!is.na(r)]) - 1L })
  indptr <- c(0L, cumsum(lengths(tag_rows)))

  dataset <- list(
    name = size, K = d$K, K_ref = ds$K_ref, M = sp$M, r_seed = sp$seed,
    generating_values = as.list(ds$truth),
    description = paste("Synthetic GWAS summary statistics. Rows are in the order of upstream's merged data frame",
                        "(= order of the input summary data). betahat = Z/sqrt(N) and varbetahat = 1/N exactly as",
                        "upstream preprocessing() computes them. Nstar/ldscore come from a synthetic LD table with",
                        "K_ref reference SNPs rs1..rsK_ref of which K are in the GWAS."),
    SNP = df$SNP, Z = as.numeric(df$Z), N = as.numeric(df$N),
    betahat = d$betahat, varbetahat = d$varbetahat, ldscore = d$ldscore, Nstar = d$Nstar,
    extremes_rows0 = match(paste0("rs", c(ds$ext_pos, ds$forced_pos[1])), df$SNP) - 1L,
    tagging_gwas_rows0 = list(
      description = "CSR neighbour lists restricted to GWAS SNPs: for row k, indices[indptr[k]:indptr[k+1]] are 0-based row indices (self included, ascending). Symmetric.",
      indptr = indptr, indices = unlist(tag_rows)),
    reference = if (size == "small") list(
      description = "the synthetic LD table handed to upstream through data(); TaggingSNPs uses upstream's comma-joined rsID format (self last)",
      SNPname = ds$dataLD$SNPname, Nstar = ds$dataLD$Nstar, LD.score.correct = ds$dataLD$LD.score.correct,
      CHR = ds$dataLD$CHR, BP = ds$dataLD$BP, TaggingSNPs = ds$dataLD$TaggingSNPs,
      in_gwas = ds$dataLD$SNPname %in% df$SNP) else list(
      description = "reference table omitted for size; K_ref reference SNPs rs1..rsK_ref, GWAS SNPs listed in dataset$SNP",
      n_reference_snps = ds$K_ref)
  )

  # ---- function-level cases --------------------------------------------------
  .oracle_msg("  function-level cases, 2-component")
  f2 <- list()
  for (ip in seq_along(pars2)) for (c0 in c0s2) {
    if (size == "small") {
      ws <- if (c0 != 20L || ip == 1) "full" else "first60"
      ms <- if (ip == 1) (if (c0 == 20L) "first60" else "full") else "none"
      ps <- TRUE
    } else { ws <- "first20"; ms <- "none"; ps <- (c0 == 10L) }
    f2[[length(f2) + 1L]] <- case_m2(pars2[[ip]], c0, d, size, ws, ms, ps)
  }
  .oracle_msg("  function-level cases, 3-component")
  f3 <- list()
  for (ip in seq_along(pars3)) for (c0 in c0s3) {
    if (size == "small") {
      ws <- if (c0 == 3L || ip == 1) "full" else "first60"
      ms <- if (ip == 1) (if (c0 == 3L) "full" else "first60") else "none"
      ps <- TRUE
    } else { ws <- "first20"; ms <- "none"; ps <- (c0 == 10L) }
    f3[[length(f3) + 1L]] <- case_m3(pars3[[ip]], c0, d, size, ws, ms, ps)
  }
  # documented failure mode: linear-space densities underflow when long double == double
  uf_par <- c(0.9, 1e-6, 1e-6)
  uf_w <- up$weight(uf_par, d$betahat, d$varbetahat, d$ldscore, 10L, d$Nstar, NT)
  underflow_demo <- list(
    par = uf_par, c0 = 10L,
    loglikelihood = num(up$loglikelihood(uf_par, d$betahat, d$varbetahat, d$ldscore, 10L, d$Nstar, NT)),
    n_rows_with_nonfinite_weights = sum(rowSums(!is.finite(uf_w)) > 0),
    rows0_with_nonfinite_weights = I(which(rowSums(!is.finite(uf_w)) > 0) - 1L),
    Nstar_of_those_rows = I(d$Nstar[rowSums(!is.finite(uf_w)) > 0]),
    note = "PLATFORM DEPENDENT. Upstream evaluates binomial/normal densities in linear space; with 64-bit long double (arm64, MSVC) (1-pic)^Nstar underflows to 0 for large Nstar, the weights become 0/0 = NaN and the log-likelihood NaN/-Inf. With 80-bit long double (x86 Linux/macOS) the same inputs stay finite. Do not use as a parity target.")

  # ---- EM scenarios ----------------------------------------------------------
  ss <- ds$sumstats
  em2 <- list(
    em_scenario(2, "a_regular_startingpic0.01", d, ss, size, c0 = 10, startingpic = 0.01),
    em_scenario(2, "a2_regular_startingpic0.05", d, ss, size, c0 = 10, startingpic = 0.05),
    em_scenario(2, "a3_regular_startingpic0.002", d, ss, size, c0 = 10, startingpic = 0.002),
    em_scenario(2, "b_c0_3", d, ss, size, c0 = 3, startingpic = 0.01),
    em_scenario(2, "b2_c0_3_startingpic0.05", d, ss, size, c0 = 3, startingpic = 0.05),
    em_scenario(2, "c_no_stratification", d, ss, size, c0 = 10, strat = FALSE, startingpic = 0.01),
    em_scenario(2, "d1_bad_start_pic0.4_sigsq0.5", d, ss, size, c0 = 10, starting = c(0.4, 0.5, 1e-6)),
    em_scenario(2, "d2_bad_start_tiny_sigsq_large_a", d, ss, size, c0 = 10, starting = c(0.01, 1e-9, 5e-5)),
    em_scenario(2, "d3_bad_start_pic0.4", d, ss, size, c0 = 10, starting = c(0.4, 5e-5, 1e-6)),
    em_scenario(2, "f_start_at_generating_values", d, ss, size, c0 = 10, starting = c(0.01, 5e-5, 1e-6)),
    em_scenario(2, "f2_start_at_generating_values_c0_20", d, ss, size, c0 = 20, starting = c(0.01, 5e-5, 1e-6)),
    em_scenario(2, "f3_start_at_generating_values_no_stratification", d, ss, size, c0 = 10, strat = FALSE, starting = c(0.01, 5e-5, 0))
  )
  em3 <- list(
    em_scenario(3, "a_regular_startingpic0.01", d, ss, size, c0 = 10, startingpic = 0.01),
    em_scenario(3, "a2_regular_startingpic0.05", d, ss, size, c0 = 10, startingpic = 0.05),
    em_scenario(3, "b_c0_3", d, ss, size, c0 = 3, startingpic = 0.01),
    em_scenario(3, "c_no_stratification", d, ss, size, c0 = 10, strat = FALSE, startingpic = 0.01),
    em_scenario(3, "d1_bad_start_pic0.4_sig0.5", d, ss, size, c0 = 10, starting = c(0.4, 0.3, 0.5, 0.05, 1e-6)),
    em_scenario(3, "d2_bad_start_tiny_sig_large_a", d, ss, size, c0 = 10, starting = c(0.01, 1/9, 1e-9, 1e-10, 5e-5)),
    em_scenario(3, "e1_p1_0.49_near_upper_bound", d, ss, size, c0 = 10, starting = c(0.01, 0.49, 1e-4, 5e-5, 1e-6)),
    em_scenario(3, "e2_p1_0.6_above_upper_bound", d, ss, size, c0 = 10, starting = c(0.01, 0.6, 1e-4, 5e-5, 1e-6)),
    em_scenario(3, "e3_pic_0.6_above_upper_bound", d, ss, size, c0 = 10, starting = c(0.6, 0.2, 1e-5, 1e-6, 1e-6)),
    em_scenario(3, "f_start_two_clusters", d, ss, size, c0 = 10, starting = c(0.01, 1/9, 2.5e-4, 2.5e-5, 1e-6)),
    em_scenario(3, "f2_start_two_clusters_c0_5", d, ss, size, c0 = 5, starting = c(0.01, 1/9, 2.5e-4, 2.5e-5, 1e-6)),
    em_scenario(3, "f3_start_two_clusters_no_stratification", d, ss, size, c0 = 10, strat = FALSE, starting = c(0.01, 1/9, 2.5e-4, 2.5e-5, 0))
  )

  fx <- list(
    meta = oracle_meta(up, upstream, threads = NT, extra = list(
      fixture = sprintf("synthetic '%s' dataset: function-level outputs and EM traces of the genuine upstream code", size))),
    dataset = dataset,
    m2 = list(parameter_layout = "par = (pic, sigmasq, a)", function_cases = f2, underflow_demo = underflow_demo, em_scenarios = em2),
    m3 = list(parameter_layout = "par = (pic, p1, sigmasq1, sigmasq2, a)", function_cases = f3, em_scenarios = em3)
  )
  write_json(fx, file.path(out_dir, sprintf("synthetic_%s.json.gz", size)))
}
clear_data_overrides(up)
