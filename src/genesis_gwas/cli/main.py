"""Command-line interface: ``genesis <command> ...``.

Commands
--------
``fit``        read summary statistics, run QC, fit M2 and/or M3, write results
``project``    expected discoveries / variance explained at future sample sizes
``predict``    expected performance of polygenic risk scores
``posterior``  per-SNP posterior null probability and shrunken effects
``qq``         model-based QQ plot with simulation envelope
``plot``       effect-size density and projection curves
``ld``         build an LD reference from PLINK genotypes
``data``       fetch / list / verify reference data
``swarm``      write a Biowulf swarm file for many traits
``demo``       run the packaged chromosome-22 example
``doctor``     report the environment and self-test the CPU/GPU backends

Run ``genesis <command> --help`` for the options of each command.
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from collections.abc import Sequence
from typing import Any

from genesis_gwas._version import __version__
from genesis_gwas.exceptions import GenesisError, GenesisWarning

__all__ = ["build_parser", "main"]

log = logging.getLogger("genesis_gwas")


def _float_list(text: str) -> list[float]:
    try:
        return [float(x) for x in text.replace(";", ",").split(",") if x.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected comma-separated numbers, got {text!r}") from exc


def _col_or_number(text: str) -> str | float:
    try:
        return float(text)
    except ValueError:
        return text


def _add_logging_options(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("logging")
    g.add_argument("-v", "--verbose", action="store_true", help="more detailed log messages")
    g.add_argument("-q", "--quiet", action="store_true", help="only warnings and errors")
    g.add_argument("--debug", action="store_true", help="show Python tracebacks on errors")


def _add_compute_options(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("compute")
    g.add_argument(
        "--device",
        default="auto",
        help="'auto' (all visible CUDA GPUs if PyTorch+CUDA is available, else CPU), 'cpu', "
        "'cuda', 'cuda:0', or a comma-separated list such as 'cuda:0,cuda:1' [auto]",
    )
    g.add_argument(
        "--threads",
        type=int,
        default=None,
        help="CPU threads [$GENESIS_NUM_THREADS, else $SLURM_CPUS_PER_TASK, else the CPUs "
        "available to this process]",
    )
    g.add_argument("--chunk-size", type=int, default=None, help="SNPs per chunk [automatic]")
    g.add_argument(
        "--backend",
        choices=["numpy", "torch"],
        default=None,
        help="force a backend (default: numpy on CPU, torch on GPU)",
    )


def _add_reference_options(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("LD reference")
    g.add_argument(
        "--reference",
        default="eur_1kg_hm3",
        help="reference name ('eur_1kg_hm3', 'mini_chr22') or path to a bundle directory "
        "[eur_1kg_hm3]",
    )
    g.add_argument("--ld-window-kb", type=float, default=None, help="LD window in kb [1000]")
    g.add_argument("--ld-r2", type=float, default=None, help="tagging r^2 threshold [0.1]")
    g.add_argument(
        "--data-dir", default=None, help="where reference bundles are stored [$GENESIS_DATA_DIR]"
    )
    g.add_argument(
        "--no-download", action="store_true", help="never download; fail if files are missing"
    )


def _add_sumstats_options(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("summary-statistics columns (auto-detected unless given)")
    g.add_argument("--snp", help="SNP identifier column")
    g.add_argument("--z", help="Z-statistic column")
    g.add_argument("--beta", help="effect column (used with --se or --p)")
    g.add_argument("--se", help="standard-error column")
    g.add_argument("--or", dest="odds_ratio", help="odds-ratio column")
    g.add_argument("--p", help="P-value column")
    g.add_argument(
        "--n", type=_col_or_number, help="sample-size column, or one number for all SNPs"
    )
    g.add_argument("--n-cases", type=_col_or_number, help="case-count column or number")
    g.add_argument("--n-controls", type=_col_or_number, help="control-count column or number")
    g.add_argument(
        "--case-fraction", type=float, help="fraction of cases; converts a total N to effective N"
    )
    g.add_argument(
        "--n-scale",
        type=float,
        default=1.0,
        help="multiply N by this factor (0.25 for METAL/PGC Neff)",
    )
    g.add_argument(
        "--n-from-se",
        action="store_true",
        help="per-SNP effective N = 1/(2f(1-f)SE^2); needs SE and EAF",
    )
    g.add_argument("--chr", dest="chrom", help="chromosome column")
    g.add_argument("--pos", help="base-pair position column")
    g.add_argument("--eaf", help="effect-allele-frequency column")
    g.add_argument("--info", help="imputation-quality column")
    g.add_argument("--sep", help="field delimiter [detected]")
    m = p.add_argument_group("matching and quality control")
    m.add_argument("--match-on", choices=["auto", "rsid", "position"], default="auto")
    m.add_argument("--build", help="genome build of the positions (GRCh37 or GRCh38) [detected]")
    m.add_argument(
        "--no-qc", action="store_true", help="skip the recommended filters (original filter=FALSE)"
    )
    m.add_argument(
        "--n-min-fraction",
        type=float,
        default=0.67,
        help="keep N >= this x 90th percentile of N [0.67]",
    )
    m.add_argument(
        "--max-chisq", type=float, default=80.0, help="drop SNPs with Z^2 above this [80]"
    )
    m.add_argument(
        "--keep-mhc", action="store_true", help="do not restrict to the HapMap3-without-MHC list"
    )
    m.add_argument(
        "--min-info",
        type=float,
        default=None,
        help="minimum imputation INFO (if the column exists)",
    )
    m.add_argument(
        "--min-maf", type=float, default=None, help="minimum MAF (if a frequency column exists)"
    )


def _add_model_options(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("model and algorithm")
    g.add_argument(
        "--model",
        choices=["2", "3", "both"],
        default="both",
        help="2 = single normal, 3 = two-normal mixture, both = fit both and select by BIC [both]",
    )
    g.add_argument("--no-stratification", action="store_true", help="fix the inflation term a = 0")
    g.add_argument("--c0", type=int, default=10, help="truncation level of latent counts [10]")
    g.add_argument(
        "--c0-max",
        type=int,
        default=20,
        help="ceiling for raising c0 when truncation matters at the solution; "
        "set equal to --c0 for a fixed truncation level [20]",
    )
    g.add_argument(
        "--tail-tol",
        type=float,
        default=1e-6,
        help="tolerated mean truncated prior mass before c0 is raised [1e-6]",
    )
    g.add_argument(
        "--starting", type=_float_list, help="explicit starting values (3 numbers for M2, 5 for M3)"
    )
    g.add_argument(
        "--starting-pic",
        type=_float_list,
        help="starting proportion(s) of susceptibility SNPs; several values = multi-start [0.01]",
    )
    g.add_argument("--max-iter", type=int, default=3000, help="maximum EM iterations [3000]")
    g.add_argument("--newton-steps", type=int, default=2, help="Newton steps per M-step [2]")
    g.add_argument(
        "--tol-scale",
        type=float,
        default=1.0,
        help="multiply all convergence tolerances by this factor (e.g. 0.01 = 100x tighter) [1]",
    )
    g.add_argument(
        "--bic-gamma", type=float, default=0.5, help="tuning constant of the modified BIC [0.5]"
    )
    g.add_argument(
        "--M",
        dest="M",
        type=int,
        default=None,
        help="number of reference SNPs [size of the reference]",
    )
    g.add_argument(
        "--legacy",
        action="store_true",
        help="reproduce the numerical behaviour of the original R/C++ package exactly",
    )
    g.add_argument(
        "--sandwich-double-count",
        action="store_true",
        help="count each marker's own score twice in the sandwich estimator (as the original does)",
    )
    g.add_argument(
        "--no-variance", action="store_true", help="skip standard errors and information criteria"
    )
    g.add_argument(
        "--no-polish",
        action="store_true",
        help="report the EM stopping point instead of the Newton-polished maximiser",
    )
    g.add_argument(
        "--no-accelerate",
        action="store_true",
        help="run the EM to its own stopping rule before Newton's method (slower, same result)",
    )
    g.add_argument(
        "--newton-after",
        type=int,
        default=50,
        help="EM iterations before the first Newton attempt [50]",
    )
    b = p.add_argument_group("binary traits")
    b.add_argument(
        "--population-prevalence", type=float, help="disease prevalence in the population"
    )
    b.add_argument("--sample-prevalence", type=float, help="proportion of cases in the GWAS")


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser (exposed for documentation and tests)."""
    parser = argparse.ArgumentParser(
        prog="genesis",
        description="GENESIS: GENetic Effect-Size distribution Inference from Summary-level data.",
        epilog="Documentation: https://jeyabbalas.github.io/GENESIS/",
    )
    parser.add_argument("--version", action="version", version=f"genesis-gwas {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    # ------------------------------------------------------------------ fit
    p = sub.add_parser(
        "fit",
        help="fit the effect-size distribution to GWAS summary statistics",
        description="Read summary statistics, apply QC, merge with the LD reference, fit the "
        "two- and/or three-component model and write estimates, standard errors and diagnostics.",
    )
    p.add_argument("sumstats", help="summary-statistics file (text, optionally gzipped)")
    p.add_argument("--out", "-o", required=True, help="output prefix (files PREFIX.*)")
    p.add_argument(
        "--print-every", type=int, default=50, help="log every N-th EM iteration, 0 = never [50]"
    )
    p.add_argument("--qq", action="store_true", help="also write a model-based QQ plot")
    p.add_argument("--qq-sims", type=int, default=100, help="simulations for the QQ envelope [100]")
    p.add_argument(
        "--posterior", action="store_true", help="also write per-SNP posterior summaries"
    )
    p.add_argument(
        "--project",
        type=_float_list,
        metavar="N1,N2,...",
        help="also project discoveries at these sample sizes",
    )
    p.add_argument("--seed", type=int, default=123, help="random seed for simulations [123]")
    _add_sumstats_options(p)
    _add_reference_options(p)
    _add_model_options(p)
    _add_compute_options(p)
    _add_logging_options(p)
    p.set_defaults(func="fit")

    # -------------------------------------------------------------- project
    p = sub.add_parser("project", help="project future GWAS discoveries from a fitted model")
    p.add_argument("fit", help="result JSON written by `genesis fit`")
    p.add_argument(
        "--n", type=_float_list, required=True, help="future sample sizes, comma-separated"
    )
    p.add_argument("--alpha", type=float, default=5e-8, help="significance level [5e-8]")
    p.add_argument("--ci", action="store_true", help="add simulation-based confidence intervals")
    p.add_argument("--n-sim", type=int, default=1000, help="simulations for the intervals [1000]")
    p.add_argument("--coverage", type=float, default=0.95, help="interval coverage [0.95]")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--out", "-o", help="write a TSV here (default: print)")
    _add_logging_options(p)
    p.set_defaults(func="project")

    # -------------------------------------------------------------- predict
    p = sub.add_parser(
        "predict", help="expected AUC of polygenic risk scores at given sample sizes"
    )
    p.add_argument("fit", help="result JSON written by `genesis fit`")
    p.add_argument(
        "--n", type=_float_list, required=True, help="training sample sizes, comma-separated"
    )
    p.add_argument(
        "--threshold",
        choices=["optimum", "gwas"],
        default="optimum",
        help="SNP-inclusion threshold: optimum or genome-wide significance [optimum]",
    )
    p.add_argument("--alpha-gwas", type=float, default=5e-8)
    p.add_argument(
        "--k-fold", type=_float_list, default=[3.0, 4.0, 5.0], help="risk multiples [3,4,5]"
    )
    p.add_argument("--out", "-o", help="write a TSV here (default: print)")
    _add_logging_options(p)
    p.set_defaults(func="predict")

    # ------------------------------------------------------------ posterior
    p = sub.add_parser(
        "posterior", help="per-SNP posterior null probability and shrunken effect sizes"
    )
    p.add_argument("sumstats")
    p.add_argument(
        "--fit", required=True, dest="fit_json", help="result JSON written by `genesis fit`"
    )
    p.add_argument("--out", "-o", required=True, help="output TSV (gzip if it ends in .gz)")
    p.add_argument(
        "--ignore-inflation",
        action="store_true",
        help="ignore the inflation term a, as the original postmarginal() does",
    )
    _add_sumstats_options(p)
    _add_reference_options(p)
    _add_compute_options(p)
    _add_logging_options(p)
    p.set_defaults(func="posterior")

    # ------------------------------------------------------------------- qq
    p = sub.add_parser("qq", help="QQ plot of observed P-values against the fitted model")
    p.add_argument("sumstats")
    p.add_argument(
        "--fit", required=True, dest="fit_json", help="result JSON written by `genesis fit`"
    )
    p.add_argument(
        "--out", "-o", required=True, help="output prefix (PREFIX.qq.png and PREFIX.qq.tsv.gz)"
    )
    p.add_argument("--n-sim", type=int, default=100, help="number of simulations [100]")
    p.add_argument("--coverage", type=float, default=0.8, help="envelope coverage [0.8]")
    p.add_argument("--axis-max", type=float, default=10.0, help="axis limit in -log10(P) [10]")
    p.add_argument("--seed", type=int, default=123)
    _add_sumstats_options(p)
    _add_reference_options(p)
    _add_logging_options(p)
    p.set_defaults(func="qq")

    # ----------------------------------------------------------------- plot
    p = sub.add_parser(
        "plot", help="plot effect-size densities or projection curves for fitted models"
    )
    p.add_argument(
        "kind",
        choices=["density", "discoveries", "variance"],
        help="density of joint effects, expected discoveries, or %% heritability explained",
    )
    p.add_argument("fits", nargs="+", help="result JSON files (optionally LABEL=path)")
    p.add_argument("--out", "-o", required=True, help="image file (.png, .pdf, .svg)")
    p.add_argument(
        "--n-max", type=float, default=1e6, help="largest sample size for projection curves [1e6]"
    )
    p.add_argument(
        "--xlim", type=float, default=0.02, help="half-width of the effect-size axis [0.02]"
    )
    _add_logging_options(p)
    p.set_defaults(func="plot")

    # ------------------------------------------------------------------- ld
    p = sub.add_parser("ld", help="build an LD reference bundle from PLINK genotypes")
    ldsub = p.add_subparsers(dest="ld_command", metavar="<ld command>")
    b = ldsub.add_parser(
        "build",
        help="compute N*, LD scores and tagging neighbourhoods for a reference panel",
        description="Build a reference bundle (same format as the bundled European reference) "
        "from PLINK .bed/.bim/.fam files of any population.",
    )
    b.add_argument(
        "--bfile",
        required=True,
        help="PLINK prefix; use {chrom} for per-chromosome files, e.g. 1000G.AFR.{chrom}",
    )
    b.add_argument("--out", "-o", required=True, help="output bundle directory")
    b.add_argument("--name", required=True, help="name of the new reference")
    b.add_argument(
        "--snps", help="file with the SNP identifiers to keep (e.g. HapMap3), one per line"
    )
    b.add_argument("--keep", help="file with the sample IDs to keep")
    b.add_argument("--window-kb", type=_float_list, default=[1000.0], help="window(s) in kb [1000]")
    b.add_argument("--r2", type=_float_list, default=[0.1], help="tagging threshold(s) [0.1]")
    b.add_argument("--min-maf", type=float, default=0.05, help="minimum MAF in the panel [0.05]")
    b.add_argument(
        "--n-null",
        type=int,
        default=0,
        help="LD-correlated null Z replicates to store, for QQ envelopes [0]",
    )
    b.add_argument(
        "--null-statistic",
        choices=["t", "normal", "projection"],
        default="t",
        help="how null Z-scores are computed from null phenotypes [t]",
    )
    b.add_argument(
        "--round-r2",
        type=int,
        default=None,
        help="round r^2 to this many significant digits before thresholding "
        "(6 reproduces the original reference tables exactly) [no rounding]",
    )
    b.add_argument(
        "--bundle-version", default="1.0", help="version label recorded in the bundle [1.0]"
    )
    b.add_argument("--population", default="unknown")
    b.add_argument("--genome-build", default="GRCh37")
    b.add_argument("--seed", type=int, default=123)
    _add_compute_options(b)
    _add_logging_options(b)
    b.set_defaults(func="ld_build")

    # ----------------------------------------------------------------- data
    p = sub.add_parser("data", help="manage reference data")
    dsub = p.add_subparsers(dest="data_command", metavar="<data command>")
    f = dsub.add_parser("fetch", help="download reference files (resumable, checksum-verified)")
    f.add_argument("bundle", nargs="?", default="eur_1kg_hm3")
    f.add_argument(
        "--components",
        default="default",
        help="comma-separated: default, ld, pairld, nullz, examples, all, or an LD "
        "configuration id such as w2000kb_r0.05 [default]",
    )
    f.add_argument(
        "--dest", dest="data_dir", default=None, help="destination directory [$GENESIS_DATA_DIR]"
    )
    _add_logging_options(f)
    f.set_defaults(func="data_fetch")
    li = dsub.add_parser("list", help="list available references and what is installed")
    li.add_argument("--data-dir", default=None)
    _add_logging_options(li)
    li.set_defaults(func="data_list")
    ve = dsub.add_parser("verify", help="re-check the checksums of a bundle directory")
    ve.add_argument("path", help="bundle directory")
    _add_logging_options(ve)
    ve.set_defaults(func="data_verify")

    # ---------------------------------------------------------------- swarm
    p = sub.add_parser(
        "swarm",
        help="write a Biowulf swarm file that fits many traits",
        description="Create a swarm file (one `genesis fit` command per summary-statistics file) "
        "and print the `swarm` command to submit it. Nothing is submitted. Everything after a "
        "literal `--` is passed on to every `genesis fit` command, e.g. "
        "`genesis swarm *.tsv.gz --outdir results -- --model both --n-cases NCASE --n-controls NCONTROL`.",
    )
    p.add_argument("sumstats", nargs="+", help="summary-statistics files (one job each)")
    p.add_argument("--outdir", required=True, help="directory for results")
    p.add_argument(
        "--swarmfile", default="genesis.swarm", help="swarm file to write [genesis.swarm]"
    )
    p.add_argument("--gpu", action="store_true", help="request one GPU per job")
    p.add_argument("--cpus", type=int, default=16, help="CPUs per job [16]")
    p.add_argument("--mem-gb", type=int, default=16, help="memory per job in GB [16]")
    p.add_argument("--time", default="04:00:00", help="walltime per job [04:00:00]")
    p.add_argument(
        "--activate", default="", help="shell line that activates the environment in each job"
    )
    _add_logging_options(p)
    p.set_defaults(func="swarm", fit_args=[])

    # ----------------------------------------------------------------- demo
    p = sub.add_parser("demo", help="run the packaged chromosome-22 example end to end")
    p.add_argument(
        "--out", "-o", default=None, help="output prefix (default: a temporary directory)"
    )
    _add_compute_options(p)
    _add_logging_options(p)
    p.set_defaults(func="demo")

    # --------------------------------------------------------------- doctor
    p = sub.add_parser(
        "doctor",
        help="report the environment and self-test the compute backends",
        description="Print versions, the CPUs/GPUs visible to this process, and verify that every "
        "available backend/device reproduces the reference results on the packaged test data.",
    )
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.add_argument(
        "--benchmark",
        action="store_true",
        help="also time one pass over a synthetic genome-wide data set on every backend/device "
        "(use it inside a job to decide between CPUs and GPUs and to set --time)",
    )
    p.add_argument(
        "--benchmark-snps",
        type=int,
        default=1_000_000,
        help="number of synthetic SNPs for --benchmark [1000000]",
    )
    _add_compute_options(p)
    _add_logging_options(p)
    p.set_defaults(func="doctor")
    return parser


def _configure_logging(args: argparse.Namespace, logfile: str | None = None) -> None:
    level = (
        logging.WARNING
        if getattr(args, "quiet", False)
        else (logging.DEBUG if getattr(args, "verbose", False) else logging.INFO)
    )
    root = logging.getLogger("genesis_gwas")
    # The logger passes everything; each handler applies its own threshold, so that
    # --quiet silences the terminal without emptying the log file.
    root.setLevel(logging.DEBUG)
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(fmt)
    stream.setLevel(level)
    root.addHandler(stream)
    if logfile:
        from pathlib import Path

        Path(logfile).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logfile, mode="w")
        fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s %(message)s"))
        fh.setLevel(logging.DEBUG if getattr(args, "verbose", False) else logging.INFO)
        root.addHandler(fh)
    root.propagate = False

    def showwarning(
        message: Any, category: Any, filename: Any, lineno: Any, file: Any = None, line: Any = None
    ) -> None:
        if isinstance(message, GenesisWarning) or issubclass(category, GenesisWarning):
            root.warning("WARNING: %s", message)
        else:
            root.debug("%s:%s: %s: %s", filename, lineno, category.__name__, message)

    warnings.showwarning = showwarning
    warnings.simplefilter("default", GenesisWarning)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of the ``genesis`` command. Returns the process exit status."""
    parser = build_parser()
    raw = list(sys.argv[1:] if argv is None else argv)
    passthrough: list[str] = []
    if raw and raw[0] == "swarm" and "--" in raw:
        # `genesis swarm FILES... [options] -- <options for every genesis fit>`
        cut = raw.index("--")
        raw, passthrough = raw[:cut], raw[cut + 1 :]
    args = parser.parse_args(raw)
    args.argv = [*raw, *(["--", *passthrough] if passthrough else [])]
    if passthrough:
        args.fit_args = passthrough
    if not getattr(args, "func", None):
        parser.print_help()
        return 2
    from genesis_gwas.cli import commands

    handler = getattr(commands, f"cmd_{args.func}")
    try:
        return int(handler(args) or 0)
    except GenesisError as exc:
        if getattr(args, "debug", False):
            raise
        if log.handlers:
            log.error("ERROR: %s", exc)
        else:
            print(f"genesis: error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:  # pragma: no cover - interactive
        print("genesis: interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
