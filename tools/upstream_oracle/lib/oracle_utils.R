# oracle_utils.R -- shared helpers for the fixture generators.
#
# JSON conventions used by every fixture written here
#   * numbers are IEEE-754 doubles printed with 17 significant digits
#     (jsonlite digits = I(17)), which round-trips every double exactly
#     (jsonlite's digits = NA keeps only 15 digits and does NOT round-trip);
#   * non-finite numbers are written as the strings "NaN", "Inf", "-Inf", "NA";
#   * matrices are arrays of rows (row-major);
#   * length-1 vectors are written as scalars unless wrapped with I().

suppressPackageStartupMessages(requireNamespace("jsonlite"))

oracle_meta <- function(up, upstream_dir, threads = 1L, extra = list()) {
  git <- function(...) tryCatch(
    system2("git", c("-C", shQuote(upstream_dir), ...), stdout = TRUE, stderr = FALSE),
    error = function(e) NA_character_, warning = function(w) NA_character_)
  cxx <- tryCatch(system2("clang++", "--version", stdout = TRUE, stderr = FALSE)[1],
                  error = function(e) NA_character_)
  if (is.na(cxx)) cxx <- tryCatch(system2("c++", "--version", stdout = TRUE, stderr = FALSE)[1],
                                  error = function(e) NA_character_)
  ld_digits <- .Machine$longdouble.digits
  c(list(
    generator = "tools/upstream_oracle (genuine upstream code; see README.md)",
    upstream_repo = "https://github.com/yandorazhang/GENESIS",
    upstream_commit = git("rev-parse", "HEAD")[1],
    upstream_commit_date = git("log", "-1", "--format=%ad", "--date=short")[1],
    em_em_cpp_sha256_pristine = unname(tryCatch(
      strsplit(system2("shasum", c("-a", "256", shQuote(file.path(upstream_dir, "src", "EM_EM.cpp"))),
                       stdout = TRUE), " ")[[1]][1], error = function(e) NA_character_)),
    build_patches = I("patches/EM_EM.cpp.patch (removes the `//[[Rcpp::plugins(openmp)]]` directive; build-only)"),
    r_version = R.version.string,
    platform = R.version$platform,
    os = paste(Sys.info()[c("sysname", "release", "machine")], collapse = " "),
    compiler = cxx,
    rcpp_version = as.character(utils::packageVersion("Rcpp")),
    rcpparmadillo_version = as.character(utils::packageVersion("RcppArmadillo")),
    sizeof_long_double = .Machine$sizeof.longdouble,
    long_double_mantissa_bits = if (is.null(ld_digits)) .Machine$double.digits else ld_digits,
    long_double_note = if (is.null(ld_digits) || ld_digits == .Machine$double.digits)
      "long double == double (64-bit) on this platform; x86 builds use 80-bit extended precision and can differ in the last few digits"
      else "extended-precision long double",
    openmp = up$.oracle$openmp$mode,
    openmp_note = up$.oracle$openmp$note,
    num_threads = threads,
    date = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"),
    json_conventions = "doubles written with 17 significant digits (exact round-trip); non-finite values are the strings NaN/Inf/-Inf/NA; matrices are row-major arrays of rows"
  ), extra)
}

write_json <- function(x, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  txt <- jsonlite::toJSON(x, digits = I(17), auto_unbox = TRUE, null = "null", na = "string",
                          matrix = "rowmajor", pretty = FALSE, force = TRUE)
  if (grepl("\\.gz$", path)) {
    con <- gzfile(path, open = "wb", compression = 9)
    writeLines(txt, con, useBytes = TRUE)
    close(con)   # before measuring the size
  } else {
    writeLines(txt, path, useBytes = TRUE)
  }
  .oracle_msg(sprintf("wrote %s (%.1f kB)", path, file.info(path)$size / 1024))
  invisible(path)
}

# 1x1 matrices / named scalars -> plain numbers.
num <- function(x) as.numeric(x)
# Keep names of a named atomic vector as a JSON object.
named <- function(x) as.list(x)
# Plain (unnamed) matrix.
mat <- function(x) { x <- as.matrix(x); dimnames(x) <- NULL; x }

# Parse the per-iteration lines upstream prints from EM_func / EM_func3
# (print = TRUE): "i, llk, par..., heritability, c0, t_w, t_pi, t_var, ".
# Values are printed by Rcout with 6 significant digits only.
parse_em_print <- function(printed) {
  rows <- grep("^[0-9]+, ", printed, value = TRUE)
  if (!length(rows)) return(NULL)
  m <- do.call(rbind, lapply(strsplit(rows, ",\\s*"), function(p) suppressWarnings(as.numeric(p))))
  m <- m[, colSums(!is.na(m)) > 0, drop = FALSE]
  unname(m)
}

# Convert the locals captured from a genuine genesis() call into a JSON-ready
# list.  `per_snp`: "none" (sums only), "head" (first n_head rows) or "full".
genesis_locals_to_list <- function(cap, per_snp = c("none", "head", "full"), n_head = 50,
                                   keep_snp_order = TRUE) {
  per_snp <- match.arg(per_snp)
  L <- cap$locals
  comp <- L$modelcomponents
  getn <- function(nm) if (!is.null(L[[nm]])) num(L[[nm]]) else NULL
  est_list <- L$estimates
  # numeric/matrix members of the returned `estimates` list are stored under
  # their own keys below; keep the list itself verbatim as well (names are the
  # labels a user of upstream sees).
  est_json <- lapply(est_list, function(v) {
    if (is.matrix(v)) mat(v) else if (is.list(v)) lapply(v, function(u) if (is.matrix(u)) mat(u) else u) else v
  })
  rows <- switch(per_snp, none = integer(0), head = seq_len(min(n_head, nrow(L$m_S))), full = seq_len(nrow(L$m_S)))
  out <- list(
    modelcomponents = comp,
    K = L$K, M = L$M, N_SNPs_summary = L$N_SNPs_summary,
    c0_final = num(L$c0),
    BIC_gamma = L$BIC.gamma, stratification = L$stratification,
    tolerances = list(pic = L$tolerance_pic, p1 = L$tolerance_p1, sigsq1 = L$tolerance_sigsq1,
                      sigsq2 = L$tolerance_sigsq2, a = L$tolerance_a, llk = L$tolerance_llk,
                      maxEM = L$maxEM, steps = L$steps),
    starting = num(L$starting),
    fit = num(L$fit),
    fit_layout = if (comp == 2)
      "i_last, llk_final, pic, sigmasq, a, heritability, c0, t_weight, t_pi, t_var (timings are CPU seconds; ignore)"
      else "i_last, llk_final, pic, p1, sigmasq1, sigmasq2, a, heritability, c0, t_weight, t_pi, t_var (timings are CPU seconds; ignore)",
    est = num(L$est),
    llk = num(L$llk),
    mean_n = mean(L$n),
    m_I = mat(L$m_I), J = mat(L$J), inv_I = mat(L$inv_I),
    var_est = mat(L$var_est), sd_est = num(L$sd_est),
    ds = num(L$ds), aic = num(L$aic), bic = num(L$bic),
    causalnum = num(L$causalnum), sd_causalnum = num(L$sd_causalnum),
    heritability = num(L$heritability), sd_heritability = num(L$sd_heritability),
    risk = num(L$risk), sd_risk = num(L$sd_risk),
    largenum = getn("largenum"), sd_largenum = getn("sd_largenum"),
    heritlarge = getn("heritlarge"), sd_heritlarge = getn("sd_heritlarge"),
    heritsmall = getn("heritsmall"), sd_heritsmall = getn("sd_heritsmall"),
    m_S_colsums = num(colSums(L$m_S)), m_Sbar_colsums = num(colSums(L$m_Sbar)),
    m_S_rows = if (length(rows)) mat(L$m_S[rows, , drop = FALSE]) else NULL,
    m_Sbar_rows = if (length(rows)) mat(L$m_Sbar[rows, , drop = FALSE]) else NULL,
    per_snp_rows = per_snp,
    estimates_returned = est_json,
    printed_trace = cap$printed,
    printed_trace_matrix = parse_em_print(cap$printed),
    snp_order = if (keep_snp_order) as.character(L$SNP) else NULL,
    snp_order_md5 = digest_chr(as.character(L$SNP))
  )
  out[!vapply(out, is.null, logical(1))]
}

# md5 of a character vector joined by "\n" (no trailing newline).
digest_chr <- function(x) {
  f <- tempfile(); on.exit(unlink(f))
  con <- file(f, open = "wb"); writeChar(paste(x, collapse = "\n"), con, eos = NULL, useBytes = TRUE); close(con)
  unname(tools::md5sum(f))
}

# LD-score-regression starting values for the 2-component model, exactly as
# written in upstream R/genesis.R (lines 113-136), including the NA-llk retry.
ldsc_start_m2 <- function(up, betahat, varbetahat, ldscore, Nstar, M, c0, startingpic = NA, cores = 1L) {
  chisq <- (betahat / sqrt(varbetahat))^2
  tem <- ldscore * mean(1 / varbetahat) / M
  ld.fit <- lm(chisq ~ tem)
  starting <- rep(0, 3)
  if (is.na(startingpic)) { starting[1] <- 0.01 }
  if (!is.na(startingpic)) { starting[1] <- startingpic }
  starting[2] <- ld.fit$coefficients[2] / (M * starting[1])
  starting[3] <- (ld.fit$coefficients[1] - 1) / M
  if (starting[2] < 0) { starting[2] <- 1e-5; starting[3] <- 2e-6 }
  llk <- up$loglikelihood(starting, betahat, varbetahat, ldscore, c0, Nstar, cores)
  if (is.na(llk)) {
    if (is.na(startingpic)) { starting[1] <- 5e-3 }
    if (!is.na(startingpic)) { starting[1] <- startingpic }
    starting[2] <- ld.fit$coefficients[2] / (M * starting[1])
    starting[3] <- (ld.fit$coefficients[1] - 1) / M
    if (starting[2] < 0) { starting[2] <- 1e-5; starting[3] <- 2e-6 }
  }
  list(starting = starting, ldsc_intercept = unname(ld.fit$coefficients[1]),
       ldsc_slope = unname(ld.fit$coefficients[2]), llk_at_start = llk)
}

quiet <- function(expr) { utils::capture.output(v <- force(expr)); v }

# polyriskpredict() returns list(alpha, AUC, PPI, PCI) where alpha/AUC may carry
# stray names from optimize() and PPI/PCI are named by k.fold.
prp_json <- function(x) list(alpha = unname(as.numeric(x$alpha)), AUC = unname(as.numeric(x$AUC)),
                             PPI = as.list(x$PPI), PCI = as.list(x$PCI))
