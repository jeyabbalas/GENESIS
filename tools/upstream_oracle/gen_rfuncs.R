#!/usr/bin/env Rscript
# gen_rfuncs.R -- fixtures for upstream's pure-R functions (no EM involved):
# h2transfer, numInterval, dmixssnp, projection, polyriskpredict, postmarginal,
# preprocessing; plus seeded one-off runs of the RNG-based functions
# (projection with CI, futuregc) labelled "stochastic".
#
# Usage:
#   Rscript --vanilla gen_rfuncs.R <tools_dir> <upstream_dir> <work_dir> <out_dir> [futuregc=yes|no]

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4) stop("usage: gen_rfuncs.R <tools_dir> <upstream_dir> <work_dir> <out_dir> [futuregc=yes|no]")
tools <- args[1]; upstream <- args[2]; work <- args[3]; out_dir <- args[4]
do_futuregc <- !(length(args) >= 5 && args[5] == "no")
.libPaths(c(file.path(work, "rlib"), .libPaths()))
source(file.path(tools, "lib", "build_upstream.R"))
source(file.path(tools, "lib", "oracle_utils.R"))
up <- build_upstream(upstream, work, tools, openmp = "auto")

# parameter vectors printed in upstream examples/examples.md (height GWAS)
est2 <- c(4.906797e-03, 5.733005e-05, 1.861491e-06)
v2 <- matrix(c(2.870170e-07, -2.426575e-09, -6.077209e-11, -2.426575e-09, 2.491765e-11, 4.715283e-13,
               -6.077209e-11, 4.715283e-13, 1.720162e-14), 3, 3)
est3 <- c(8.899809e-03, 9.476025e-02, 1.458650e-04, 2.227118e-05, 1.567643e-06)
v3 <- matrix(c(1.327856e-06, -1.131049e-05, 6.912489e-10, -2.901301e-09, -9.388865e-11,
               -8.568269e-06, 3.380985e-04, -1.543479e-07, 1.771036e-09, 8.989668e-10,
               -1.846216e-09, -1.353542e-07, 1.517545e-10, 1.166563e-11, 1.901686e-14,
               -3.068113e-09, 9.520417e-09, 5.633801e-12, 9.427492e-12, 1.220433e-13,
               -9.111521e-11, 1.063848e-09, -1.526281e-13, 1.042123e-13, 1.406410e-14), 5, 5)
ests <- list(m2_height = est2, m3_height = est3,
             m2_alt = c(0.02, 1.2e-5, -8e-7), m3_alt = c(0.015, 0.3, 4e-4, 3e-5, 0))

try_rec <- function(expr) {
  w <- character()
  v <- withCallingHandlers(
    tryCatch(list(status = "ok", value = expr), error = function(e) list(status = "error", error_message = conditionMessage(e))),
    warning = function(cond) { w <<- c(w, conditionMessage(cond)); invokeRestart("muffleWarning") })
  if (length(w)) v$warnings <- I(unique(w))
  v
}

# ---- h2transfer ------------------------------------------------------------
h2_in <- list(c(0.5, 0.05, 0.01, 0.5), c(0.3, 0.02, 0.12, 0.4), c(2.1, 0.095, 0.01, 0.44),
              c(0.694, 0.105, 0.08, 0.174), c(1.0, 0.1, 0.5, 0.5), c(0.1, 0.01, 0.001, 0.1))
h2 <- lapply(h2_in, function(a) c(list(h2.log = a[1], se.h2.log = a[2], population.prevalence = a[3], sample.prevalence = a[4]),
                                 up$h2transfer(a[1], a[2], a[3], a[4])))

# ---- numInterval -----------------------------------------------------------
iv <- list(c(0.005, Inf), c(0, 0.001), c(0.001, 0.005), c(0.01, 0.02), c(0, Inf), c(0.02, Inf), c(0.0005, 0.03))
numint <- list()
for (nm in names(ests)) for (a in iv) for (M in c(1070777, 500000)) {
  numint[[length(numint) + 1L]] <- list(est_name = nm, est = ests[[nm]], lower = a[1], upper = a[2], M = M,
                                        value = num(up$numInterval(a[1], a[2], ests[[nm]], M)))
}
numint_negative_lower <- lapply(c("m2_height", "m3_height"), function(nm) list(
  est_name = nm, est = ests[[nm]], lower = -0.005, upper = 0.005, M = 1070777,
  value = num(up$numInterval(-0.005, 0.005, ests[[nm]])),
  note = "lower < 0 is outside the documented use (|effect| interval); upstream then includes the point mass at zero"))

# ---- dmixssnp --------------------------------------------------------------
xg <- seq(-0.03, 0.03, length.out = 41)
dmix <- lapply(names(ests), function(nm) list(est_name = nm, est = ests[[nm]], x = xg,
                                              density = num(sapply(xg, function(t) up$dmixssnp(t, ests[[nm]]))),
                                              density_vectorised_call = num(up$dmixssnp(xg, ests[[nm]]))))

# ---- projection (CI = FALSE; deterministic) ---------------------------------
proj <- list()
for (nm in names(ests)) for (n in c(10000, 50000, 133000, 253288, 1e6, 5e6)) for (alpha in c(5e-8, 1e-5)) {
  r <- try_rec(up$projection(ests[[nm]], v = NULL, n = n, gwas.significance = alpha, CI = FALSE))
  proj[[length(proj) + 1L]] <- c(list(est_name = nm, est = ests[[nm]], n = n, gwas.significance = alpha,
                                      tol = c(1e-12, 1e-15), M = 1070777), r)
}
proj_M <- lapply(c("m2_height", "m3_height"), function(nm)
  c(list(est_name = nm, est = ests[[nm]], n = 253288, gwas.significance = 5e-8, tol = c(1e-12, 1e-15), M = 200000),
    try_rec(up$projection(ests[[nm]], v = NULL, n = 253288, M = 200000, CI = FALSE))))

# ---- polyriskpredict --------------------------------------------------------
prp <- list()
add_prp <- function(label, N, Ps, Sig2s, M, M1, type, alp = 5e-8, k.fold = 3:5) {
  r <- try_rec(prp_json(up$polyriskpredict(N = N, Ps = Ps, Sig2s = Sig2s, M = M, M1 = M1, type = type, alp.GWAS = alp, k.fold = k.fold)))
  prp[[length(prp) + 1L]] <<- c(list(label = label, N = N, Ps = Ps, Sig2s = Sig2s, M = M, M1 = M1, type = type,
                                    alp.GWAS = alp, k.fold = I(k.fold)), r)
}
for (N in c(10000, 50000, 253288, 1e6)) for (type in c("optimum", "GWAS")) {
  add_prp("one variance (2-component height est, Sig2s = sigmasq)", N, c(0.5, 0.5), c(est2[2], est2[2]), 1070777, 1070777 * est2[1], type)
  add_prp("two variances (3-component height est)", N, c(est3[2], 1 - est3[2]), c(est3[3], est3[4]), 1070777, 1070777 * est3[1], type)
}
add_prp("one variance, examples.md as written (Sig2s = est[3] = a, not sigmasq)", 253288, c(0.5, 0.5), c(est2[3], est2[3]), 1070777, 1070777 * est2[1], "optimum")
add_prp("one variance, examples.md as written (Sig2s = est[3] = a, not sigmasq)", 253288, c(0.5, 0.5), c(est2[3], est2[3]), 1070777, 1070777 * est2[1], "GWAS")
add_prp("two variances, M = 200000 independent SNPs, disease-like values", 100000, c(0.1, 0.9), c(8e-4, 8e-5), 200000, 3000, "optimum")
add_prp("two variances, M = 200000 independent SNPs, disease-like values", 100000, c(0.1, 0.9), c(8e-4, 8e-5), 200000, 3000, "GWAS", alp = 5e-8)
add_prp("GWAS threshold 1e-5, k.fold 2:6", 253288, c(est3[2], 1 - est3[2]), c(est3[3], est3[4]), 1070777, 1070777 * est3[1], "GWAS", alp = 1e-5, k.fold = 2:6)

# ---- postmarginal on a synthetic 10-column data frame ------------------------
set.seed(7701)
Kp <- 80
Nstar <- c(1, 1, 2, 3, 5, 9, 10, 11, 12, 20, 313, 710, pmin(710, pmax(1, round(exp(rnorm(Kp - 12, log(37), 0.72))))))
ld <- pmax(1, Nstar * runif(Kp, 0.15, 0.7)); ld[Nstar == 1] <- 1
N <- round(runif(Kp, 50000, 150000))
Z <- c(rnorm(Kp - 6), sqrt(79.9), -sqrt(79.9), 0, 5, -5, 3)
pm_df <- data.frame(SNP = paste0("rs", seq_len(Kp)), Z = Z, N = N, betahat = Z / sqrt(N), varbetahat = 1 / N,
                    Nstar = as.integer(Nstar), TaggingSNPs = paste0("rs", seq_len(Kp)), LD.score.correct = ld,
                    CHR = 1L, BP = 1000L * seq_len(Kp), stringsAsFactors = FALSE)
pm_cases <- list()
for (nm in names(ests)) for (c0 in c(10, 5)) {
  r <- up$postmarginal(pm_df, ests[[nm]], c0 = c0)
  pm_cases[[length(pm_cases) + 1L]] <- list(est_name = nm, est = ests[[nm]], c0 = c0,
                                            prob_beta0 = num(r$prob.beta0), posteriormean_marginal = num(r$posteriormean.marginal))
}
postmarginal <- list(
  note = "postmarginal() never uses the stratification term a (last element of est): the residual variance is varbetahat alone",
  data = list(SNP = pm_df$SNP, Z = pm_df$Z, N = pm_df$N, betahat = pm_df$betahat, varbetahat = pm_df$varbetahat,
              Nstar = pm_df$Nstar, ldscore = pm_df$LD.score.correct),
  cases = pm_cases)

# ---- preprocessing (real bundled LD table + HapMap3 list) --------------------
e <- new.env(); load(file.path(upstream, "data", "LDwindow1MB_cutoff0.1.RData"), envir = e)
load(file.path(upstream, "data", "w_hm3.noMHC.snplist.RData"), envir = e)
ldt <- e$dataLD; hm3 <- as.character(e$w_hm3.noMHC.snplist$SNP)
c22 <- ldt[ldt$CHR == 22, ]; c22 <- c22[order(c22$BP), ]
mhc <- ldt[ldt$CHR == 6 & ldt$BP > 26e6 & ldt$BP < 34e6, ]
mhc_not_in_list <- as.character(mhc$SNPname[!(mhc$SNPname %in% hm3)])
in_list_not_in_ld <- setdiff(hm3, ldt$SNPname)
snps <- c(as.character(c22$SNPname[seq(10, 300, by = 10)]),      # 30 ordinary chr22 reference SNPs
          mhc_not_in_list[1:3],                                  # in the LD table, but not in w_hm3.noMHC.snplist
          in_list_not_in_ld[1:2],                                # in the HapMap3 list, not in the LD table
          "rs999999999", "chr1:12345:A:G")                       # unknown IDs
set.seed(7702)
n_in <- round(runif(length(snps), 90000, 100000)); n_in[c(4, 9)] <- c(30000, 60001)   # below 0.67 * 90th percentile
z_in <- round(rnorm(length(snps)), 6); z_in[2] <- 9.1; z_in[6] <- sqrt(80); z_in[7] <- 8.9442719; z_in[12] <- -8.95
pp_in <- data.frame(snp = snps, z = z_in, n = n_in, stringsAsFactors = FALSE)
pp_out <- function(df) list(n_rows = nrow(df), columns = colnames(df), SNP = as.character(df$SNP), Z = num(df$Z), N = num(df$N),
                            betahat = num(df$betahat), varbetahat = num(df$varbetahat), Nstar = num(df$Nstar),
                            LD.score.correct = num(df$LD.score.correct), CHR = num(df$CHR), BP = num(df$BP),
                            n_tagging = lengths(strsplit(as.character(df$TaggingSNPs), ",")))
ppA <- up$preprocessing(pp_in, LDcutoff = 0.1, LDwindow = 1, filter = TRUE)
ppB <- up$preprocessing(pp_in, LDcutoff = 0.1, LDwindow = 1, filter = FALSE)
pp_na1 <- pp_in; pp_na1$z[3] <- NA                               # exactly one NA z
pp_na2 <- pp_in; pp_na2$z[c(3, 5)] <- NA                         # two NA z
pp_nan <- pp_in; pp_nan$n[c(10, 11)] <- NA                       # two NA n
pp_nan1 <- pp_in; pp_nan1$n[10] <- NA                            # one NA n
ppC <- up$preprocessing(pp_na1, 0.1, 1, FALSE); ppD <- up$preprocessing(pp_na2, 0.1, 1, FALSE)
ppE <- up$preprocessing(pp_nan, 0.1, 1, FALSE); ppE1 <- up$preprocessing(pp_nan1, 0.1, 1, FALSE)
ppF <- try_rec(pp_out(up$preprocessing(pp_nan, 0.1, 1, TRUE)))
ppG <- try_rec(pp_out(up$preprocessing(pp_na1, 0.1, 1, TRUE)))
preprocessing <- list(
  note = paste("Row order of the result = order of the input rows that match the LD table (base::merge, sort = FALSE).",
               "filter = TRUE keeps N >= 0.67 * quantile(N, 0.9) (R type-7 quantile over ALL input rows), then SNPs in",
               "w_hm3.noMHC.snplist, then Z^2 <= 80; NA rows are removed only when MORE THAN ONE value is NA",
               "(`if(length(inx)>1)`), so a single NA z or n survives into the model inputs."),
  input = list(snp = pp_in$snp, z = pp_in$z, n = pp_in$n,
               threshold_0.67_q90 = 0.67 * unname(quantile(pp_in$n, 0.9)),
               in_w_hm3_noMHC = pp_in$snp %in% hm3, in_LD_table = pp_in$snp %in% ldt$SNPname, z_squared = pp_in$z^2),
  filter_TRUE = pp_out(ppA),
  filter_FALSE = pp_out(ppB),
  one_NA_z_at_row3_filter_FALSE = pp_out(ppC),
  two_NA_z_at_rows3and5_filter_FALSE = pp_out(ppD),
  two_NA_n_at_rows10and11_filter_FALSE = pp_out(ppE),
  one_NA_n_at_row10_filter_FALSE = pp_out(ppE1),
  two_NA_n_filter_TRUE = ppF,
  one_NA_z_filter_TRUE = ppG)
rm(e, ldt, hm3, c22, mhc); gc()

# ---- stochastic (seeded single runs; NOT reproducible outside R) -------------
qnamed <- function(x) unname(as.numeric(x))
proj_ci <- function(r) list(heritability = num(r$heritability), Numdiscoveries_est_low_high = qnamed(r$Numdiscoveries),
                            pheno.variance_est_low_high = qnamed(r$pheno.variance), GVpercentage_est_low_high = qnamed(r$GVpercentage))
pc2 <- try_rec(proj_ci(up$projection(est2, v2, n = 253288, CI = TRUE)))
pc3 <- try_rec(proj_ci(up$projection(est3, v3, n = 253288, CI = TRUE)))
stochastic <- list(
  label = "stochastic",
  note = "Seeded single runs of upstream's RNG-based functions. They depend on R's RNG stream and are only suitable for statistical sanity comparisons.",
  projection_CI_m2_height = c(list(est = est2, v = mat(v2), n = 253288, nsim = 1000, CI.coverage = 0.95, seeds = 123), pc2),
  projection_CI_m3_height = c(list(est = est3, v = mat(v3), n = 253288, nsim = 1000, CI.coverage = 0.95, seeds = 123,
                                   v_note = "v3 is the asymmetric matrix printed in examples.md; MASS::mvrnorm uses eigen(symmetric = TRUE), i.e. only its lower triangle"),
                              pc3),
  projection_CI_negative_a = c(list(est = ests$m2_alt, v = mat(v2), n = 253288),
                               try_rec(proj_ci(up$projection(ests$m2_alt, v2, n = 253288, CI = TRUE)))),
  projection_CI_negative_a_note = paste("est with a < 0: projection() takes log(est) for the bootstrap, so the `a` slot is NaN",
                                        "(warning 'NaNs produced'); the run still completes because the power integrals never use a.",
                                        "(est and v are deliberately mismatched here; only the behaviour matters.)"),
  projection_CI_a_zero = c(list(est = c(0.005, 5e-5, 0), v = mat(v2), n = 253288),
                           try_rec(proj_ci(up$projection(c(0.005, 5e-5, 0), v2, n = 253288, CI = TRUE)))),
  projection_CI_a_zero_note = "a == 0 exactly (what upstream returns with stratification = FALSE): log(0) = -Inf and 1/est = Inf enter the bootstrap covariance"
)
if (do_futuregc) {
  .oracle_msg("futuregc (full-genome simulation; slow) ...")
  t0 <- Sys.time(); g2 <- up$futuregc(est2, n = 253288, nsim = 1, M = 1070777, seeds = 123); t2 <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  t0 <- Sys.time(); g3 <- up$futuregc(est3, n = 253288, nsim = 1, M = 1070777, seeds = 123); t3 <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  t0 <- Sys.time(); g2b <- up$futuregc(est2, n = 133000, nsim = 2, M = 1070777, seeds = 123); t2b <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  stochastic$futuregc <- list(
    list(est = est2, n = 253288, nsim = 1, seeds = 123, lambdaGC = num(g2), seconds = t2),
    list(est = est3, n = 253288, nsim = 1, seeds = 123, lambdaGC = num(g3), seconds = t3,
         note = "the 3-component branch calls set.seed(iter*123) and ignores the seeds argument"),
    list(est = est2, n = 133000, nsim = 2, seeds = 123, lambdaGC = num(g2b), seconds = t2b,
         note = "mean over 2 simulations; the observed lambdaGC of the bundled height GWAS (n ~ 133k) is the natural comparison"))
}

fx <- list(
  meta = oracle_meta(up, upstream, threads = 1L, extra = list(fixture = "pure-R upstream functions")),
  estimates_used = ests,
  h2transfer = h2,
  numInterval = numint, numInterval_negative_lower = numint_negative_lower,
  dmixssnp = dmix,
  projection = proj, projection_other_M = proj_M,
  polyriskpredict = prp,
  postmarginal = postmarginal,
  preprocessing = preprocessing,
  stochastic = stochastic)
write_json(fx, file.path(out_dir, "rfuncs.json"))
