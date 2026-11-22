"""Implementations of the ``genesis`` sub-commands."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import sys
import tempfile
import time
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from genesis_gwas._version import __version__
from genesis_gwas.cli.main import _configure_logging
from genesis_gwas.core.derived import DERIVED_LABELS
from genesis_gwas.exceptions import GenesisError, InputError

log = logging.getLogger("genesis_gwas")


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------
def _reader_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    keys = (
        "snp",
        "z",
        "beta",
        "se",
        "odds_ratio",
        "p",
        "n",
        "n_cases",
        "n_controls",
        "case_fraction",
        "n_scale",
        "n_from_se",
        "chrom",
        "pos",
        "eaf",
        "info",
        "sep",
    )
    return {k: getattr(args, k) for k in keys if getattr(args, k, None) not in (None, False)}


def _load_reference(args: argparse.Namespace) -> Any:
    from genesis_gwas.data.reference import load_reference

    return load_reference(
        args.reference,
        window_kb=args.ld_window_kb,
        r2_min=args.ld_r2,
        data_dir=args.data_dir,
        download=not args.no_download,
    )


def _qc_config(args: argparse.Namespace) -> Any:
    from genesis_gwas.data.prepare import QCConfig

    if args.no_qc:
        return QCConfig.none()
    return QCConfig(
        n_min_fraction=args.n_min_fraction
        if args.n_min_fraction and args.n_min_fraction > 0
        else None,
        max_chisq=args.max_chisq if args.max_chisq and args.max_chisq > 0 else None,
        hm3_no_mhc=not args.keep_mhc,
        exclude_mhc=not args.keep_mhc,
        min_info=args.min_info,
        min_maf=args.min_maf,
    )


def _prepare(args: argparse.Namespace, ref: Any) -> Any:
    from genesis_gwas.data.prepare import prepare

    return prepare(
        args.sumstats,
        ref,
        qc=_qc_config(args),
        match_on=args.match_on,
        build=args.build,
        M=getattr(args, "M", None),
        **_reader_kwargs(args),
    )


def _plan(args: argparse.Namespace) -> Any:
    from genesis_gwas.compute import resolve_plan

    return resolve_plan(
        args.device, threads=args.threads, chunk_size=args.chunk_size, backend=args.backend
    )


def _em_config(args: argparse.Namespace) -> Any:
    from genesis_gwas.core.em import EMConfig

    base = EMConfig(
        c0=args.c0,
        c0_max=max(args.c0_max, args.c0),
        tail_tol=args.tail_tol,
        max_iter=args.max_iter,
        newton_steps=args.newton_steps,
        legacy=args.legacy,
    )
    s = args.tol_scale
    if s != 1.0:
        if not s > 0:
            raise InputError("--tol-scale must be > 0")
        base = replace(
            base,
            tol_pic=base.tol_pic * s,
            tol_p1=base.tol_p1 * s,
            tol_sigmasq=base.tol_sigmasq * s,
            tol_a=base.tol_a * s,
            tol_loglik=base.tol_loglik * s,
        )
    return base


def _progress(print_every: int) -> Any:
    if print_every <= 0:
        return None

    def cb(rec: dict[str, Any]) -> None:
        if rec["iter"] % print_every == 0:
            theta = "  ".join(f"{v:.5e}" for v in rec["theta"])
            log.info(
                "  %-12s iter %5d  logLik %.4f  c0 %d  [%s]",
                rec["stage"],
                rec["iter"],
                rec["loglik"],
                rec["c0"],
                theta,
            )

    return cb


def _estimates_table(fits: list[Any], selected: Any) -> str:
    """Tidy long-format table: one row per model x quantity."""
    rows = ["model\tselected\tquantity\tlabel\testimate\tse"]
    for fit in fits:
        sel = "1" if fit is selected else "0"
        for name, est, se in zip(fit.param_names, fit.estimates, fit.se, strict=True):
            rows.append(f"{fit.model}\t{sel}\t{name}\t{name}\t{est:.10g}\t{se:.6g}")
        for key, val in fit.derived.items():
            rows.append(
                f"{fit.model}\t{sel}\t{key}\t{DERIVED_LABELS.get(key, key)}\t{val.estimate:.10g}\t{val.se:.6g}"
            )
        for key in ("loglik", "bic", "aic", "ds"):
            rows.append(f"{fit.model}\t{sel}\t{key}\t{key}\t{getattr(fit, key):.12g}\tnan")
        for key, value in (
            ("n_snps", fit.n_snps),
            ("n_iter", fit.n_iter),
            ("c0", fit.c0),
            ("converged", int(fit.converged)),
        ):
            rows.append(f"{fit.model}\t{sel}\t{key}\t{key}\t{value}\tnan")
        liab = fit.diagnostics.get("liability")
        if liab:
            for scale in ("observed", "liability"):
                rows.append(
                    f"{fit.model}\t{sel}\theritability_{scale}\tHeritability ({scale} scale)\t"
                    f"{liab[f'h2_{scale}']:.10g}\t{liab[f'se_h2_{scale}']:.6g}"
                )
    return "\n".join(rows) + "\n"


def _load_fit(path: str) -> Any:
    from genesis_gwas.result import FitResult

    p = Path(path)
    if not p.exists():
        raise InputError(f"Result file not found: {p}")
    return FitResult.from_json(p)


def _write_table(frame: Any, out: str | None) -> None:
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out, sep="\t", index=False, float_format="%.8g")
        log.info("Wrote %s", out)
    else:
        frame.to_csv(sys.stdout, sep="\t", index=False, float_format="%.8g")


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------
def cmd_fit(args: argparse.Namespace) -> int:
    """``genesis fit``."""
    from genesis_gwas.fit import fit
    from genesis_gwas.result import select_model

    prefix = Path(args.out)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    _configure_logging(args, logfile=f"{prefix}.log")
    log.info("genesis-gwas %s", __version__)
    log.info("Command: genesis %s", " ".join(shlex.quote(str(a)) for a in args.argv))
    t0 = time.perf_counter()

    plan = _plan(args)
    ref = _load_reference(args)
    log.info("Reference: %s", ref.describe())
    data = _prepare(args, ref)

    common: dict[str, Any] = dict(
        stratification=not args.no_stratification,
        em=_em_config(args),
        variance=not args.no_variance,
        polish=False if args.no_polish else None,
        accelerate=not args.no_accelerate,
        newton_after=args.newton_after,
        bic_gamma=args.bic_gamma,
        sandwich_double_count=True if args.sandwich_double_count else None,
        population_prevalence=args.population_prevalence,
        sample_prevalence=args.sample_prevalence,
        plan=plan,
        progress=_progress(args.print_every),
    )
    models = [2, 3] if args.model == "both" else [int(args.model)]
    if args.starting is not None and args.model == "both":
        raise InputError("--starting applies to one model; combine it with --model 2 or --model 3.")
    fits: list[Any] = []
    for nc in models:
        log.info("Fitting the %s-component model ...", "two" if nc == 2 else "three")
        kwargs = dict(common)
        if args.starting is not None:
            kwargs["starting"] = args.starting
        elif nc == 3 and fits:
            kwargs["starting"] = fits[0]  # the original's recipe: start M3 from the M2 estimate
        if "starting" not in kwargs and args.starting_pic is not None:
            kwargs["starting_pic"] = args.starting_pic
        res = fit(data, nc, **kwargs)
        res.to_json(f"{prefix}.M{nc}.json")
        log.info("\n%s\n", res.summary())
        fits.append(res)

    selected = fits[0]
    report = [
        f"genesis-gwas {__version__}",
        f"input: {args.sumstats}",
        f"reference: {ref.describe()}",
        "",
    ]
    report += [f.summary() + "\n" for f in fits]
    if len(fits) > 1 and not args.no_variance:
        cmp = select_model(fits)
        selected = cmp.best
        report.append(cmp.summary())
        log.info("\n%s", cmp.summary())
    report += ["", "Quality control"]
    report += [
        f"  {s['step']:<64} removed {s['removed']:>10,}   remaining {s['remaining']:>10,}"
        for s in data.meta["qc"]
    ]
    Path(f"{prefix}.summary.txt").write_text("\n".join(report) + "\n")
    Path(f"{prefix}.estimates.tsv").write_text(_estimates_table(fits, selected))
    Path(f"{prefix}.best.json").write_text(selected.to_json(include_history=False) + "\n")

    if args.project:
        from genesis_gwas.analysis.projection import project

        frame = project(
            selected, np.asarray(args.project, dtype=float), ci=not args.no_variance, seed=args.seed
        ).to_frame()
        _write_table(frame, f"{prefix}.projection.tsv")
    if args.posterior:
        from genesis_gwas.analysis.posterior import posterior_marginal

        post = posterior_marginal(data, selected, plan=plan)
        post.to_csv(f"{prefix}.posterior.tsv.gz", sep="\t", index=False, float_format="%.6g")
        log.info("Wrote %s", f"{prefix}.posterior.tsv.gz")
    if args.qq:
        _run_qq(
            data,
            selected,
            ref,
            f"{prefix}",
            n_sim=args.qq_sims,
            coverage=0.8,
            seed=args.seed,
            axis_max=10.0,
        )

    log.info(
        "Selected model: %s. Results written to %s.*  (%.1f s)",
        selected.model,
        prefix,
        time.perf_counter() - t0,
    )
    return 0


def _run_qq(
    data: Any,
    fit: Any,
    ref: Any,
    prefix: str,
    *,
    n_sim: int,
    coverage: float,
    seed: int,
    axis_max: float,
) -> None:
    from genesis_gwas.analysis.simulate import qq_simulation
    from genesis_gwas.plotting import new_axes, qq_plot, save_figure

    qq = qq_simulation(data, fit, ref, n_sim=n_sim, coverage=coverage, seed=seed)
    ax = qq_plot(qq, new_axes(), axis_max=axis_max, title=f"{fit.model}")
    save_figure(ax, f"{prefix}.qq.png")
    qq.thin(20000).to_frame().to_csv(
        f"{prefix}.qq.tsv.gz", sep="\t", index=False, float_format="%.6g"
    )
    log.info(
        "Wrote %s.qq.png and %s.qq.tsv.gz (lambda_obs = %.3f, lambda_fit = %.3f; %d simulations%s)",
        prefix,
        prefix,
        qq.lambda_obs,
        qq.lambda_mean,
        qq.n_sim,
        "" if qq.ld_errors else "; independent errors, envelope too narrow",
    )


# ---------------------------------------------------------------------------
# downstream analyses
# ---------------------------------------------------------------------------
def cmd_project(args: argparse.Namespace) -> int:
    """``genesis project``."""
    from genesis_gwas.analysis.projection import project

    _configure_logging(args)
    fit = _load_fit(args.fit)
    res = project(
        fit,
        np.asarray(args.n, dtype=float),
        alpha=args.alpha,
        ci=args.ci,
        n_sim=args.n_sim,
        coverage=args.coverage,
        seed=args.seed,
    )
    _write_table(res.to_frame(), args.out)
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    """``genesis predict``."""
    import pandas as pd

    from genesis_gwas.analysis.prediction import polyrisk_predict

    _configure_logging(args)
    fit = _load_fit(args.fit)
    rows = []
    for n in args.n:
        res = polyrisk_predict(
            n,
            fit,
            M=args.M,
            threshold=args.threshold,
            alpha_gwas=args.alpha_gwas,
            k_fold=tuple(args.k_fold),
        )
        row: dict[str, Any] = {
            "n": n,
            "threshold": args.threshold,
            "alpha": float(res.alpha),
            "auc": float(res.auc),
        }
        for k, ppi, pci in zip(
            args.k_fold, np.atleast_1d(res.ppi), np.atleast_1d(res.pci), strict=True
        ):
            row[f"pop_frac_at_{k:g}x_risk"] = float(ppi)
            row[f"case_frac_at_{k:g}x_risk"] = float(pci)
        rows.append(row)
    _write_table(pd.DataFrame(rows), args.out)
    return 0


def cmd_posterior(args: argparse.Namespace) -> int:
    """``genesis posterior``."""
    from genesis_gwas.analysis.posterior import posterior_marginal

    _configure_logging(args)
    fit = _load_fit(args.fit_json)
    ref = _load_reference(args)
    args.M = fit.M
    data = _prepare(args, ref)
    frame = posterior_marginal(
        data, fit, include_inflation=not args.ignore_inflation, plan=_plan(args)
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, sep="\t", index=False, float_format="%.6g")
    log.info("Wrote %s (%s SNPs)", args.out, f"{len(frame):,}")
    return 0


def cmd_qq(args: argparse.Namespace) -> int:
    """``genesis qq``."""
    _configure_logging(args)
    fit = _load_fit(args.fit_json)
    ref = _load_reference(args)
    args.M = fit.M
    data = _prepare(args, ref)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    _run_qq(
        data,
        fit,
        ref,
        args.out,
        n_sim=args.n_sim,
        coverage=args.coverage,
        seed=args.seed,
        axis_max=args.axis_max,
    )
    return 0


def cmd_plot(args: argparse.Namespace) -> int:
    """``genesis plot``."""
    from genesis_gwas.plotting import new_axes, plot_effect_density, plot_projection, save_figure

    _configure_logging(args)
    fits: dict[str, Any] = {}
    for item in args.fits:
        label, _, path = item.rpartition("=")
        fits[label or Path(path).name.removesuffix(".json")] = _load_fit(path)
    ax = new_axes((6.4, 4.4))
    if args.kind == "density":
        plot_effect_density(fits, ax, xlim=(-args.xlim, args.xlim))
    else:
        grid = np.linspace(args.n_max / 200.0, args.n_max, 200)
        plot_projection(
            fits, ax, n_grid=grid, what="discoveries" if args.kind == "discoveries" else "gv"
        )
    save_figure(ax, args.out)
    log.info("Wrote %s", args.out)
    return 0


# ---------------------------------------------------------------------------
# LD reference
# ---------------------------------------------------------------------------
def cmd_ld_build(args: argparse.Namespace) -> int:
    """``genesis ld build``."""
    from genesis_gwas.ld import build_reference

    _configure_logging(args)
    configs = [(w, r) for w in args.window_kb for r in args.r2]
    device = "cpu" if args.device == "auto" else args.device
    out = build_reference(
        args.bfile,
        args.out,
        name=args.name,
        snps=args.snps,
        keep=args.keep,
        configs=configs,
        min_maf=args.min_maf,
        n_null=args.n_null,
        null_statistic=args.null_statistic,
        round_r2=args.round_r2,
        seed=args.seed,
        population=args.population,
        genome_build=args.genome_build,
        version=args.bundle_version,
        device=device,
        backend=args.backend,
        threads=args.threads,
    )
    log.info("Reference bundle written to %s; use it with --reference %s", out, out)
    return 0


# ---------------------------------------------------------------------------
# data management
# ---------------------------------------------------------------------------
def cmd_data_fetch(args: argparse.Namespace) -> int:
    """``genesis data fetch``."""
    from genesis_gwas.data.download import fetch_bundle

    _configure_logging(args)
    comps = [c.strip() for c in args.components.split(",") if c.strip()]
    root = fetch_bundle(args.bundle, comps if len(comps) > 1 else comps[0], data_dir=args.data_dir)
    log.info("Reference %s is ready in %s", args.bundle, root)
    print(root)
    return 0


def cmd_data_list(args: argparse.Namespace) -> int:
    """``genesis data list``."""
    from genesis_gwas.data.download import known_bundles, load_packaged_manifest
    from genesis_gwas.data.reference import default_data_dir, list_packaged_references

    _configure_logging(args)
    base = Path(args.data_dir).expanduser() if args.data_dir else default_data_dir()
    print(f"Data directory: {base}")
    for name in known_bundles():
        manifest = load_packaged_manifest(name)
        root = base / name / str(manifest["version"])
        files = manifest["files"]
        have = [rel for rel in files if (root / rel).exists()]
        size = sum(files[r]["bytes"] for r in have)
        total = sum(v["bytes"] for v in files.values())
        print(f"\n{name} v{manifest['version']}: {manifest['panel']}")
        print(
            f"  installed {len(have)}/{len(files)} files ({size / 1e6:.0f} of {total / 1e6:.0f} MB) in {root}"
        )
        print("  LD configurations: " + ", ".join(c["id"] for c in manifest["ld_configs"]))
        for rel in sorted(files):
            mark = "x" if rel in have else " "
            print(f"   [{mark}] {rel:<48} {files[rel]['bytes'] / 1e6:8.1f} MB")
    print("\nPackaged (no download needed): " + ", ".join(list_packaged_references()))
    return 0


def cmd_data_verify(args: argparse.Namespace) -> int:
    """``genesis data verify``."""
    from genesis_gwas.data.download import verify_bundle

    _configure_logging(args)
    status = verify_bundle(args.path)
    bad = 0
    for rel, st in status.items():
        print(f"{st:8s} {rel}")
        bad += st == "corrupt"
    if bad:
        raise GenesisError(f"{bad} file(s) failed verification; delete them and fetch again.")
    return 0


# ---------------------------------------------------------------------------
# HPC helpers
# ---------------------------------------------------------------------------
def cmd_swarm(args: argparse.Namespace) -> int:
    """``genesis swarm``."""
    from genesis_gwas.hpc.swarm import write_swarm

    _configure_logging(args)
    options: dict[str, Any] = dict(
        outdir=args.outdir,
        swarmfile=args.swarmfile,
        fit_args=list(args.fit_args),
        gpu=args.gpu,
        cpus=args.cpus,
        mem_gb=args.mem_gb,
        walltime=args.time,
        activate=args.activate,
        data_dir=args.data_dir,
        bundle=args.bundle,
        partition=args.partition,
        gpu_constraint=args.gpu_constraint,
    )
    command = write_swarm(args.sumstats, **options)
    dry_run = command.replace("swarm ", "swarm --devel ", 1)
    if not (args.data_dir or os.environ.get("GENESIS_DATA_DIR")):
        log.warning(
            "WARNING: GENESIS_DATA_DIR is not set and --data-dir was not given: the jobs will "
            "look for the reference data in the default cache in your home directory."
        )
    log.info("Wrote %s with %d fit(s).", args.swarmfile, len(args.sumstats))
    log.info("Dry run (submits nothing):  %s", dry_run)
    log.info("Submit it from the Biowulf login node with:")
    print(command)
    return 0


# ---------------------------------------------------------------------------
# demo and doctor
# ---------------------------------------------------------------------------
def cmd_demo(args: argparse.Namespace) -> int:
    """``genesis demo``."""
    import genesis_gwas as gg

    _configure_logging(args)
    tmp = None
    if args.out is None:
        tmp = tempfile.mkdtemp(prefix="genesis-demo-")
        prefix = Path(tmp) / "demo"
    else:
        prefix = Path(args.out)
        prefix.parent.mkdir(parents=True, exist_ok=True)
    plan = _plan(args)
    ref = gg.load_reference("mini_chr22")
    log.info("Demo: GIANT height GWAS (Lango Allen et al. 2010), chromosome 22 only.")
    log.info("Reference: %s", ref.describe())
    data = gg.prepare(ref.example_path(), ref)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", gg.exceptions.GenesisWarning)
        m2 = gg.fit(data, 2, plan=plan)
        m3 = gg.fit(data, 3, starting=m2, plan=plan)
    cmp = gg.select_model([m2, m3])
    for res in (m2, m3):
        res.to_json(f"{prefix}.M{res.n_components}.json")
        print(res.summary() + "\n")
    print(cmp.summary())
    print(
        "\nNote: this is a smoke test. One chromosome carries too little information to separate "
        "two clusters of effect sizes, so the three-component fit is expected to be flagged as not "
        "identified and M2 to be selected; with M = 15,633 the numbers refer to chromosome 22 alone."
    )
    print(f"\nResults: {prefix}.M2.json, {prefix}.M3.json")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    """``genesis doctor``."""
    import genesis_gwas as gg
    from genesis_gwas.backends import torch_available
    from genesis_gwas.compute import describe_environment, resolve_plan
    from genesis_gwas.core.engine import Engine
    from genesis_gwas.core.likelihood import CompositeLikelihood
    from genesis_gwas.core.spec import ModelSpec

    _configure_logging(args)
    env = describe_environment()
    env["genesis_gwas"] = __version__
    ref = gg.load_reference("mini_chr22")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = gg.prepare(ref.example_path(), ref, qc=False)
    theta = np.array([0.02, 0.1, 6e-5, 8e-6, 5e-7])
    spec = ModelSpec(3)

    def evaluate(plan: Any) -> tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        with Engine(data, plan) as eng:
            lik = CompositeLikelihood(spec, eng, 10)
            res = lik.em_pass(theta)
            si = lik.score_info(theta)
            for _ in range(3):
                lik.em_pass(theta)
        assert res.counts is not None and res.grad is not None and res.hess is not None
        vec = np.concatenate(
            [
                [res.loglik, res.loglik_unnorm],
                res.counts,
                res.grad,
                res.hess.ravel(),
                si.information.ravel(),
            ]
        )
        return vec, time.perf_counter() - t0

    candidates: list[tuple[str, dict[str, Any]]] = [
        ("numpy / cpu, 1 thread", dict(device="cpu", threads=1)),
        (
            f"numpy / cpu, {args.threads or env['available_cpus']} thread(s)",
            dict(device="cpu", threads=args.threads),
        ),
    ]
    if torch_available():
        candidates.append(
            ("torch / cpu", dict(device="cpu", backend="torch", threads=args.threads))
        )
        for dev in env["cuda_devices"]:
            candidates.append(
                (
                    f"torch / cuda:{dev['index']} ({dev['name']})",
                    dict(device=f"cuda:{dev['index']}"),
                )
            )
        if len(env["cuda_devices"]) > 1:
            candidates.append((f"torch / all {len(env['cuda_devices'])} GPUs", dict(device="cuda")))
    reference_vec = None
    checks = []
    ok = True
    for label, kw in candidates:
        try:
            vec, secs = evaluate(resolve_plan(**kw))
            if reference_vec is None:
                reference_vec = vec
            err = float(
                np.max(np.abs(vec - reference_vec) / np.maximum(np.abs(reference_vec), 1e-300))
            )
            passed = err < 1e-8
            checks.append({"backend": label, "max_rel_diff": err, "seconds": secs, "ok": passed})
        except Exception as exc:  # report, do not crash: this is a diagnostic tool
            checks.append({"backend": label, "error": f"{type(exc).__name__}: {exc}", "ok": False})
            passed = False
        ok = ok and passed
    env["self_test"] = checks
    env["ok"] = ok

    bench: list[dict[str, Any]] = []
    if args.benchmark:
        # One pass = everything an EM iteration needs twice (or a Newton step roughly once),
        # on synthetic data the size of a real analysis; three-component model, c0 = 10.
        rng = np.random.default_rng(0)
        k = max(1000, int(args.benchmark_snps))
        nstar = np.clip(rng.gamma(1.6, 29.0, k).astype(np.int64), 1, 700)
        nsamp = rng.uniform(1.0e5, 1.5e5, k)
        big = gg.ModelData(
            betahat=rng.normal(0.0, np.sqrt(1.0 / nsamp + 2e-6)),
            varbetahat=1.0 / nsamp,
            ldscore=np.maximum(1.0, nstar * rng.uniform(0.2, 0.55, k)),
            nstar=nstar,
            n=nsamp,
            M=k,
        )
        theta_b = np.array([0.01, 0.1, 1.5e-4, 2e-5, 1.5e-6])
        for label, kw in candidates[1:]:
            try:
                with Engine(big, resolve_plan(**kw)) as eng:
                    lik = CompositeLikelihood(spec, eng, 10)
                    lik.em_pass(theta_b)  # warm-up (allocations, GPU initialisation)
                    times = []
                    for _ in range(3):
                        t0 = time.perf_counter()
                        lik.em_pass(theta_b)
                        times.append(time.perf_counter() - t0)
                bench.append({"backend": label, "seconds_per_pass": min(times), "n_snps": k})
            except Exception as exc:
                bench.append({"backend": label, "error": f"{type(exc).__name__}: {exc}"})
        env["benchmark"] = bench

    if args.json:
        print(json.dumps(env, indent=2))
    else:
        print(f"genesis-gwas {__version__}")
        print(
            f"  python {env['python']}  numpy {env['numpy']}  scipy {env['scipy']}  torch {env['torch'] or 'not installed'}"
        )
        print(f"  platform: {env['platform']}")
        print(
            f"  CPUs available to this process: {env['available_cpus']} (machine: {env['os_cpu_count']})"
            + (
                f"  [Slurm job {env['slurm_job_id']}, --cpus-per-task={env['slurm_cpus_per_task']}]"
                if env["slurm_job_id"]
                else ""
            )
        )
        if env["torch"]:
            cuda = env.get("torch_cuda_build")
            print(
                f"  PyTorch CUDA build: {cuda or 'none (CPU-only build)'}; CUDA_VISIBLE_DEVICES={env['cuda_visible_devices']}"
            )
        if env["cuda_devices"]:
            for dev in env["cuda_devices"]:
                print(f"  GPU {dev['index']}: {dev['name']} ({dev['memory_gib']} GiB)")
        else:
            print("  GPUs: none visible")
        print(
            "\nSelf-test (three-component likelihood, scores and information on the packaged chr22 data):"
        )
        for c in checks:
            if "error" in c:
                print(f"  FAIL  {c['backend']:<44} {c['error']}")
            else:
                print(
                    f"  {'ok  ' if c['ok'] else 'FAIL'}  {c['backend']:<44} max rel. diff {c['max_rel_diff']:.1e}   {c['seconds']:.2f} s"
                )
        print(
            "\nAll backends agree."
            if ok
            else "\nPROBLEM: at least one backend failed or disagrees (see above)."
        )
        if bench:
            print(
                f"\nBenchmark: one pass over {bench[0].get('n_snps', 0):,} synthetic SNPs, "
                "three-component model, c0 = 10"
            )
            for b in bench:
                if "error" in b:
                    print(f"  FAIL  {b['backend']:<44} {b['error']}")
                else:
                    print(
                        f"        {b['backend']:<44} {b['seconds_per_pass'] * 1e3:9.1f} ms per pass"
                    )
            print(
                "  A default three-component fit needs on the order of 150 passes (two per EM "
                "iteration during the\n  50-iteration warm-up, then a few Newton steps), a "
                "two-component fit one tenth of the work per pass."
            )
    return 0 if ok else 1
