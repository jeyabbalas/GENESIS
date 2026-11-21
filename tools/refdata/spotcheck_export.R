#!/usr/bin/env Rscript
# Independent spot-check extract for verify_bundle.py.
#
#   Rscript --vanilla spotcheck_export.R <upstream_data_dir> <out_file> [n_random=50] [seed=20261121]
#
# For a random sample of reference SNPs (plus a few deliberately chosen edge cases) this
# writes every upstream field *by rsID*, reading the upstream objects directly. It shares
# no code path with export_rdata.R / build_bundle.py (no integer re-indexing is involved),
# so agreement between this file and the finished bundle checks the whole conversion.
#
# Record types (tab separated):
#   SNP    rsid
#   CFG    rsid  cfg_id  chr  bp  nstar  ldscore(%.17g)  TaggingSNPs(verbatim)
#   PAIR   rsid  neighbour rsids (comma joined, TaggingSNPinx order)  r2 (comma joined, %.17g)
#   NULL   rsid  iter  value(%.17g)
#   HM3    rsid  0|1
#   HEIGHT rsid  z|NA  n|NA

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("usage: Rscript --vanilla spotcheck_export.R <upstream_data_dir> <out_file> [n_random] [seed]")
}
data_dir <- normalizePath(args[1], mustWork = TRUE)
out_file <- args[2]
n_random <- if (length(args) >= 3) as.integer(args[3]) else 50L
seed <- if (length(args) >= 4) as.integer(args[4]) else 20261121L

load_one <- function(name) {
  e <- new.env()
  load(file.path(data_dir, paste0(name, ".RData")), envir = e)
  e
}

con <- file(out_file, "wb")
emit <- function(...) writeLines(paste(..., sep = "\t"), con, useBytes = TRUE)

# ---- choose SNPs from the default configuration ---------------------------
e <- load_one("LDwindow1MB_cutoff0.1")
x <- e$dataLD
snp_all <- as.character(x$SNPname)
set.seed(seed)
pick <- sample.int(nrow(x), n_random)
ord <- order(x$CHR, x$BP)
key <- paste(x$CHR, x$BP)
dup <- which(duplicated(key) | duplicated(key, fromLast = TRUE))
mhc <- which(x$CHR == 6 & x$BP > 26e6 & x$BP < 34e6)
special <- c(which.max(x$Nstar), which(x$Nstar == 1)[1], ord[1], ord[length(ord)],
             mhc[which.min(x$BP[mhc])], if (length(dup)) dup[1:min(2, length(dup))])
pick <- unique(c(pick, special[!is.na(special)]))
rsids <- snp_all[pick]
for (r in rsids) emit("SNP", r)

# ---- pairwise LD / tagging indices (rows follow LDwindow1MB_cutoff0.1) ----
ti <- load_one("TaggingSNPinx")$TaggingSNPinx
pl <- load_one("pairwiseLD")$pairwiseLD
for (k in pick) {
  emit("PAIR", snp_all[k], paste(snp_all[ti[[k]]], collapse = ","),
       paste(sprintf("%.17g", pl[[k]]), collapse = ","))
}
rm(ti, pl, e, x); invisible(gc())

# ---- every LD configuration ----------------------------------------------
for (w in c("0.5", "1", "2")) {
  for (cc in c("0.05", "0.1", "0.2")) {
    x <- load_one(sprintf("LDwindow%sMB_cutoff%s", w, cc))$dataLD
    cfg <- sprintf("w%dkb_r%s", as.integer(round(as.numeric(w) * 1000)), cc)
    m <- match(rsids, as.character(x$SNPname))
    if (anyNA(m)) stop("spot-check SNP missing from ", cfg)
    for (i in seq_along(rsids)) {
      k <- m[i]
      emit("CFG", rsids[i], cfg, x$CHR[k], x$BP[k], x$Nstar[k],
           sprintf("%.17g", x$LD.score.correct[k]), as.character(x$TaggingSNPs[k]))
    }
    rm(x); invisible(gc())
  }
}

# ---- null z replicates ----------------------------------------------------
for (it in c(1L, 2L, 10L, 11L, 57L, 100L)) {
  e <- load_one(sprintf("error_iter%d", it))
  m <- match(rsids, as.character(e$SNP))
  for (i in seq_along(rsids)) emit("NULL", rsids[i], it, sprintf("%.17g", e$error[m[i]]))
}

# ---- HapMap3 no-MHC membership and the height example ---------------------
w <- load_one("w_hm3.noMHC.snplist")$w_hm3.noMHC.snplist
for (r in rsids) emit("HM3", r, as.integer(r %in% as.character(w$SNP)))
h <- load_one("heightGWAS")$heightGWAS
m <- match(rsids, as.character(h$snp))
for (i in seq_along(rsids)) {
  if (is.na(m[i])) emit("HEIGHT", rsids[i], "NA", "NA")
  else emit("HEIGHT", rsids[i], as.character(h$z[m[i]]), as.character(h$n[m[i]]))
}
close(con)
cat("wrote", out_file, "for", length(rsids), "SNPs\n")
