#!/usr/bin/env Rscript
# Export the reference data bundled with upstream GENESIS (.RData files) to plain
# little-endian binaries and text files that build_bundle.py can assemble.
#
#   Rscript --vanilla export_rdata.R <upstream_data_dir> <raw_out_dir>
#
# <upstream_data_dir> is the data/ directory of https://github.com/yandorazhang/GENESIS
# (commit e4e6894d2e11ece6715d0572b0febceff1480f85) or of
# https://github.com/yandorazhang/GENESISasian (commit 4df90ac6ef3ea84219110187ef4ef5b5a79b83e6).
# LD configurations that the directory does not contain are skipped (GENESISasian ships
# only LDwindow1MB_cutoff0.1). Base R only; nothing is installed.
#
# Output layout (all binaries little-endian; *.i32 = int32, *.f64 = float64):
#   ld_w{0.5,1,2}MB_c{0.05,0.1,0.2}/
#       snpname.txt    one rsID per line, upstream dataLD row order
#       chr.i32 bp.i32 nstar.i32 ldscore.f64
#       tag_len.i32    number of tagging SNPs listed for each row
#       tag_idx0.i32   concatenated tagging lists as 0-based upstream row indices,
#                      in the order they appear in the TaggingSNPs string
#   pair_w1MB_c0.1/    TaggingSNPinx / pairwiseLD (rows = LDwindow1MB_cutoff0.1 order)
#       len.i32 tagidx0.i32 r2.f64
#   nullz/             error_iter1..100: snpname.txt + error_iterNNN.f64
#   height/heightGWAS.tsv   SNP, Z, N exactly as the upstream factor labels
#   hm3/w_hm3.noMHC.snplist.tsv   SNP, A1, A2
#   export_log.txt     provenance and per-object summaries

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
  stop("usage: Rscript --vanilla export_rdata.R <upstream_data_dir> <raw_out_dir>")
}
data_dir <- normalizePath(args[1], mustWork = TRUE)
out_dir <- args[2]
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
out_dir <- normalizePath(out_dir, mustWork = TRUE)

log_path <- file.path(out_dir, "export_log.txt")
log_con <- file(log_path, "wt")
say <- function(...) {
  msg <- paste0(...)
  cat(msg, "\n", sep = "")
  cat(msg, "\n", sep = "", file = log_con)
  flush(log_con)
}

write_bin <- function(x, path, what = c("int", "double")) {
  what <- match.arg(what)
  con <- file(path, "wb")
  on.exit(close(con))
  n <- length(x)
  chunk <- 2^24
  i <- 1
  while (i <= n) {
    j <- min(n, i + chunk - 1)
    if (what == "int") {
      v <- x[i:j]
      if (anyNA(v)) stop("NA in integer export: ", path)
      writeBin(as.integer(v), con, size = 4L, endian = "little")
    } else {
      writeBin(as.double(x[i:j]), con, size = 8L, endian = "little")
    }
    i <- j + 1
  }
  invisible(n)
}

load_one <- function(name) {
  path <- file.path(data_dir, paste0(name, ".RData"))
  e <- new.env()
  load(path, envir = e)
  e
}

md5_of <- function(name) {
  unname(tools::md5sum(file.path(data_dir, paste0(name, ".RData"))))
}

say("# export_rdata.R")
say("R: ", R.version.string, " | platform: ", R.version$platform)
say("data_dir: ", data_dir)
say("started: ", format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z"))

# --------------------------------------------------------------------------
# 1. LD tables (9 window x cutoff configurations)
# --------------------------------------------------------------------------
windows <- c("0.5", "1", "2")
cutoffs <- c("0.05", "0.1", "0.2")
for (w in windows) {
  for (cc in cutoffs) {
    name <- sprintf("LDwindow%sMB_cutoff%s", w, cc)
    if (!file.exists(file.path(data_dir, paste0(name, ".RData")))) {
      say(name, " not present: skipped")
      next
    }
    t0 <- proc.time()[3]
    e <- load_one(name)
    if (!identical(ls(e), "dataLD")) stop("unexpected objects in ", name, ": ", paste(ls(e), collapse = ","))
    x <- e$dataLD
    need <- c("SNPname", "Nstar", "TaggingSNPs", "LD.score.correct", "CHR", "BP")
    if (!all(need %in% names(x))) stop("missing columns in ", name)
    od <- file.path(out_dir, sprintf("ld_w%sMB_c%s", w, cc))
    dir.create(od, showWarnings = FALSE)

    snp <- as.character(x$SNPname)
    if (anyDuplicated(snp)) stop("duplicated SNP names in ", name)
    writeLines(snp, file.path(od, "snpname.txt"), useBytes = TRUE)
    write_bin(x$CHR, file.path(od, "chr.i32"), "int")
    write_bin(x$BP, file.path(od, "bp.i32"), "int")
    write_bin(x$Nstar, file.path(od, "nstar.i32"), "int")
    write_bin(x$LD.score.correct, file.path(od, "ldscore.f64"), "double")

    tg <- strsplit(as.character(x$TaggingSNPs), ",", fixed = TRUE)
    len <- lengths(tg)
    flat <- unlist(tg, use.names = FALSE)
    rm(tg)
    n_empty <- sum(!nzchar(flat))
    idx <- match(flat, snp)
    n_na <- sum(is.na(idx))
    if (n_na > 0 || n_empty > 0) {
      stop(name, ": ", n_na, " tagging names not found among SNPname and ", n_empty, " empty names")
    }
    write_bin(len, file.path(od, "tag_len.i32"), "int")
    write_bin(idx - 1L, file.path(od, "tag_idx0.i32"), "int")
    say(sprintf("%s md5=%s rows=%d nnz=%d len==Nstar:%s maxlen=%d sum(LD)=%.10f secs=%.0f",
                name, md5_of(name), nrow(x), length(idx), all(len == x$Nstar), max(len),
                sum(x$LD.score.correct), proc.time()[3] - t0))
    rm(e, x, flat, idx, len, snp)
    invisible(gc())
  }
}

# --------------------------------------------------------------------------
# 2. Pairwise r^2 and tagging indices for the 1 Mb / r^2 >= 0.1 configuration
# --------------------------------------------------------------------------
od <- file.path(out_dir, "pair_w1MB_c0.1")
dir.create(od, showWarnings = FALSE)
e <- load_one("TaggingSNPinx")
ti <- e$TaggingSNPinx
len_ti <- lengths(ti)
write_bin(len_ti, file.path(od, "len.i32"), "int")
write_bin(unlist(ti, use.names = FALSE) - 1, file.path(od, "tagidx0.i32"), "int")
say(sprintf("TaggingSNPinx md5=%s rows=%d nnz=%.0f", md5_of("TaggingSNPinx"), length(ti), sum(len_ti)))
rm(e, ti); invisible(gc())
e <- load_one("pairwiseLD")
pl <- e$pairwiseLD
len_pl <- lengths(pl)
if (!identical(len_pl, len_ti)) stop("pairwiseLD and TaggingSNPinx row lengths differ")
r2 <- unlist(pl, use.names = FALSE)
write_bin(r2, file.path(od, "r2.f64"), "double")
say(sprintf("pairwiseLD md5=%s rows=%d nnz=%d min=%.17g max=%.17g", md5_of("pairwiseLD"),
            length(pl), length(r2), min(r2), max(r2)))
rm(e, pl, r2); invisible(gc())

# --------------------------------------------------------------------------
# 3. Null (LD-correlated) z-score replicates: error_iter1..100
# --------------------------------------------------------------------------
od <- file.path(out_dir, "nullz")
dir.create(od, showWarnings = FALSE)
snp1 <- NULL
for (i in 1:100) {
  name <- sprintf("error_iter%d", i)
  e <- load_one(name)
  if (!setequal(ls(e), c("error", "SNP"))) stop("unexpected objects in ", name)
  s <- as.character(e$SNP)
  if (i == 1) {
    snp1 <- s
    writeLines(s, file.path(od, "snpname.txt"), useBytes = TRUE)
  } else if (!identical(s, snp1)) {
    stop("SNP order of ", name, " differs from error_iter1")
  }
  if (length(e$error) != length(s)) stop("length mismatch in ", name)
  write_bin(e$error, file.path(od, sprintf("error_iter%03d.f64", i)), "double")
  say(sprintf("%s md5=%s n=%d mean=%.10f sd=%.10f", name, md5_of(name), length(e$error),
              mean(e$error), sd(e$error)))
  rm(e)
}
invisible(gc())

# --------------------------------------------------------------------------
# 4. Example GWAS (height, Lango Allen et al. 2010) and HapMap3 no-MHC list
# --------------------------------------------------------------------------
od <- file.path(out_dir, "height")
dir.create(od, showWarnings = FALSE)
e <- load_one("heightGWAS")
h <- e$heightGWAS
if (!identical(names(h), c("snp", "z", "n"))) stop("unexpected heightGWAS columns")
out <- data.frame(SNP = as.character(h$snp), Z = as.character(h$z), N = as.character(h$n),
                  stringsAsFactors = FALSE)
con <- file(file.path(od, "heightGWAS.tsv"), "wb")  # binary mode: always "\n" line ends
write.table(out, con, sep = "\t", quote = FALSE, row.names = FALSE, col.names = TRUE)
close(con)
say(sprintf("heightGWAS md5=%s rows=%d classes=%s", md5_of("heightGWAS"), nrow(h),
            paste(vapply(h, function(v) class(v)[1], ""), collapse = ",")))
rm(e, h, out)

od <- file.path(out_dir, "hm3")
dir.create(od, showWarnings = FALSE)
e <- load_one("w_hm3.noMHC.snplist")
w <- e$w_hm3.noMHC.snplist
if (!identical(names(w), c("SNP", "A1", "A2"))) stop("unexpected w_hm3.noMHC.snplist columns")
con <- file(file.path(od, "w_hm3.noMHC.snplist.tsv"), "wb")
write.table(data.frame(SNP = as.character(w$SNP), A1 = as.character(w$A1), A2 = as.character(w$A2)),
            con, sep = "\t", quote = FALSE, row.names = FALSE, col.names = TRUE)
close(con)
say(sprintf("w_hm3.noMHC.snplist md5=%s rows=%d", md5_of("w_hm3.noMHC.snplist"), nrow(w)))

say("finished: ", format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z"))
close(log_con)
