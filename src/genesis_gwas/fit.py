"""High-level model fitting: starting values → EM → sandwich variance → derived quantities."""

from __future__ import annotations

import logging
import time
import warnings
from collections.abc import Callable, Sequence
from dataclasses import asdict, replace
from typing import Any

import numpy as np

from genesis_gwas._version import __version__
from genesis_gwas.compute import ComputePlan, resolve_plan
from genesis_gwas.core.data import ModelData
from genesis_gwas.core.derived import canonicalize_m3, derived_quantities
from genesis_gwas.core.em import EMConfig, EMResult, PolishResult, newton_polish, run_em
from genesis_gwas.core.engine import Engine
from genesis_gwas.core.likelihood import CompositeLikelihood
from genesis_gwas.core.spec import ModelSpec
from genesis_gwas.core.start import ldsc_regression, m2_start, m3_start_from_m2
from genesis_gwas.core.variance import (
    Neighborhoods,
    information_criteria,
    sandwich,
    scaled_inverse,
)
from genesis_gwas.exceptions import (
    ConvergenceWarning,
    DataWarning,
    GenesisError,
    InputError,
    NumericalWarning,
)
from genesis_gwas.result import FitResult

__all__ = ["fit"]

log = logging.getLogger("genesis_gwas")

Progress = Callable[[dict[str, Any]], None]


def _as_start_list(
    spec: ModelSpec,
    starting: Any,
    starting_pic: float | Sequence[float] | None,
) -> tuple[list[np.ndarray] | None, list[float | None]]:
    """Normalise the user's starting-value options.

    Returns ``(explicit_starts, pics)``: either a list of full starting vectors, or
    ``None`` together with the list of starting proportions to try.
    """
    if starting is not None:
        if isinstance(starting, FitResult):
            if spec.n_components == 3 and starting.n_components == 2:
                return [m3_start_from_m2(starting.estimates)], [None]
            if starting.n_components != spec.n_components:
                raise InputError(
                    f"Cannot start an {spec.name} fit from an M{starting.n_components} result."
                )
            return [np.asarray(starting.estimates, dtype=float).copy()], [None]
        arr = np.asarray(starting, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.ndim != 2 or arr.shape[1] != spec.n_params:
            raise InputError(
                f"{spec.name} starting values must have {spec.n_params} entries "
                f"{spec.param_names}; got shape {np.asarray(starting).shape}."
            )
        return [spec.validate(row) for row in arr], [None] * arr.shape[0]
    if starting_pic is None:
        return None, [None]
    if isinstance(starting_pic, (int, float)):
        return None, [float(starting_pic)]
    pics = [float(p) for p in starting_pic]
    if not pics:
        raise InputError("starting_pic must not be an empty sequence")
    return None, list(pics)


def _feasible_or_none(lik: CompositeLikelihood, theta: np.ndarray) -> float | None:
    """Log-likelihood at ``theta`` or ``None`` if it cannot be evaluated there."""
    try:
        ll, _ = lik.loglik(theta)
    except InputError:
        return None
    return ll if np.isfinite(ll) else None


def _two_component_start(
    lik2: CompositeLikelihood,
    data: ModelData,
    pic: float | None,
    legacy: bool,
    notes: list[str],
) -> np.ndarray:
    """LD-score-regression start for M2, with the original's fallback logic."""
    reg = ldsc_regression(data)
    start = m2_start(reg, data.M, pic)
    if _feasible_or_none(lik2, start) is None:
        start = m2_start(reg, data.M, pic, fallback=True)
        if _feasible_or_none(lik2, start) is None:
            if legacy:
                raise GenesisError(
                    "legacy mode: the likelihood cannot be evaluated at the LD-score-regression "
                    f"starting values {start.tolist()} (the original would propagate NaN). "
                    "Supply `starting=` explicitly or run without legacy=True."
                )
            start = start.copy()
            start[2] = 0.0
            notes.append(
                "LD-score-regression starting value for `a` was infeasible "
                "(a <= -min(varbetahat)); started from a = 0 instead."
            )
    if not lik2.spec.stratification:
        start = start.copy()
        if not legacy:
            start[2] = 0.0
    return start


def _degeneracy(spec: ModelSpec, theta: np.ndarray) -> str | None:
    """Why a three-component estimate is not identified, or ``None`` if it is fine."""
    if spec.n_components != 3:
        return None
    hi = max(float(theta[2]), float(theta[3]))
    lo = min(float(theta[2]), float(theta[3]))
    if (hi - lo) <= 1e-2 * hi:
        return "the two variance components have merged (sigmasq1 = sigmasq2 to within 1%)"
    if lo <= 1e-11:
        return "one variance component has collapsed to zero"
    return None


def _em_then_newton(
    lik: CompositeLikelihood,
    start: np.ndarray,
    cfg: EMConfig,
    callback: Callable[[dict[str, Any]], None] | None,
    *,
    polish: bool,
    accelerate: bool,
    warmup: int,
) -> tuple[EMResult, PolishResult | None]:
    """Run the EM and, if requested, finish with Newton's method.

    Without acceleration this is the EM to its stopping rule followed by one polishing
    run. With acceleration the EM is run in blocks of growing length (``warmup``,
    ``2 * warmup``, ...) and Newton's method is attempted after each block; the first
    attempt that converges ends the fit, and a declined attempt costs one or two passes
    over the data before the EM simply continues from where it was.
    """
    if not polish:
        return run_em(lik, start, cfg, callback), None
    if not accelerate:
        em = run_em(lik, start, cfg, callback)
        return em, (newton_polish(lik, em.theta, cfg) if em.status != "stalled" else None)

    theta = start
    done = 0
    block = max(1, int(warmup))
    history: list[dict[str, Any]] = []
    backtracks = 0
    first = True
    pol: PolishResult | None = None
    while True:
        n = min(block, cfg.max_iter - done)
        offset = done

        def shifted(rec: dict[str, Any], offset: int = offset) -> None:
            rec["iter"] += offset
            if callback is not None:
                callback(rec)

        part = run_em(lik, theta, replace(cfg, c0=cfg.c0 if first else lik.c0, max_iter=n), shifted)
        first = False
        history.extend(part.history)
        backtracks += part.n_backtracks
        done += part.n_iter
        theta = part.theta
        em = EMResult(
            theta=part.theta,
            loglik=part.loglik,
            loglik_unnorm=part.loglik_unnorm,
            c0=part.c0,
            n_iter=done,
            converged=part.converged,
            status=part.status,
            history=history,
            n_backtracks=backtracks,
        )
        if part.status == "stalled":
            return em, pol
        pol = newton_polish(lik, part.theta, cfg)
        if part.status == "converged" or pol.status == "converged" or done >= cfg.max_iter:
            if part.status != "converged" and pol.status == "converged":
                em.status = "newton"  # the EM warm-up was ended by a successful Newton solve
            return em, pol
        if _degeneracy(lik.spec, part.theta) is not None:
            # Newton declined and the three-component fit is collapsing onto the
            # two-component model: the remaining EM iterations would only drift along a
            # flat direction of the likelihood (thousands of them in the original).
            em.status = "not_identified"
            return em, pol
        block *= 2


def fit(
    data: ModelData,
    n_components: int = 2,
    *,
    starting: Any = None,
    starting_pic: float | Sequence[float] | None = None,
    stratification: bool = True,
    c0: int = 10,
    em: EMConfig | None = None,
    neighborhoods: Neighborhoods | None = None,
    variance: bool = True,
    polish: bool | None = None,
    accelerate: bool = True,
    newton_after: int = 50,
    bic_gamma: float = 0.5,
    legacy: bool = False,
    sandwich_double_count: bool | None = None,
    population_prevalence: float | None = None,
    sample_prevalence: float | None = None,
    device: str | Sequence[str] = "auto",
    threads: int | None = None,
    chunk_size: int | None = None,
    plan: ComputePlan | None = None,
    progress: Progress | None = None,
    keep_history: bool = True,
) -> FitResult:
    r"""Fit the two- or three-component GENESIS model by composite maximum likelihood.

    Parameters
    ----------
    data
        Model inputs, normally from :func:`genesis_gwas.prepare`.
    n_components
        ``2`` — susceptibility-SNP effects follow one normal distribution (M2);
        ``3`` — a mixture of two normals, allowing a cluster of larger effects (M3).
    starting
        Starting values: a vector in the model's parameter layout, a 2-D array with one
        start per row, or a previous :class:`~genesis_gwas.result.FitResult` (an M2
        result initialises M3 with the original's rule
        ``(pic, 1/9, 5*sigmasq, sigmasq/2, a)``). By default starting values come from
        LD-score regression, and M3 is initialised from an internal M2 fit.
    starting_pic
        Starting proportion(s) of susceptibility SNPs for the LD-score-regression start
        (default 0.01). Passing several values runs one EM per value and keeps the fit
        with the highest composite log-likelihood, as the original authors recommend.
    stratification
        Estimate the variance-inflation term ``a`` (population stratification, cryptic
        relatedness). ``False`` fixes ``a = 0``.
    c0
        Truncation level of the latent counts (shortcut for ``em.c0``).
    em
        Full EM configuration; overrides ``c0`` and ``legacy`` if given.
    neighborhoods
        Tagging neighbourhoods for the sandwich estimator. Defaults to
        ``data.neighborhoods``. Without them, markers are treated as independent and the
        standard errors are too small for real GWAS data (a warning is issued).
    variance
        Compute standard errors and information criteria (one extra pass over the data).
    polish
        After the EM has met its stopping rule, take a few Newton steps to the exact
        maximiser of the composite likelihood (see
        :func:`genesis_gwas.core.em.newton_polish`). Default: ``True``, except in
        ``legacy`` mode, where the estimate is the point at which the original's EM
        stops. The EM estimate before polishing is kept in
        ``result.diagnostics["polish"]``.
    accelerate
        With polishing enabled: do not wait for the (slow) EM to meet its stopping rule.
        Newton's method is attempted after ``newton_after`` EM iterations and, if it
        declines (information matrix not positive definite, or no ascent), again after
        each further block of twice the previous length while the EM continues. The
        result is the same stationary point that the EM followed by polishing reaches —
        Newton's method is only accepted when it converges to a point with positive
        definite information and a likelihood at least as high as the EM iterate it
        started from — in a fraction of the passes over the data (on the GIANT height
        example the three-component fit needs 50 EM iterations instead of 853). Set to
        ``False`` to run the EM to its own stopping rule first.
    newton_after
        Length of the first EM block when ``accelerate`` is on.
    bic_gamma
        Tuning constant of the modified BIC, in ``[0, 1]`` (default 0.5).
    legacy
        Reproduce the numerical behaviour of the original R/C++ package, including its
        known quirks (see the documentation page "Differences from the original").
    sandwich_double_count
        Count each marker's own score twice in the sandwich "meat", as the original code
        does. Defaults to the value of ``legacy``.
    population_prevalence, sample_prevalence
        For binary traits: if both are given, heritability is also reported on the
        observed and liability scales.
    device, threads, chunk_size, plan
        Compute options; see :func:`genesis_gwas.compute.resolve_plan`.
    progress
        Called once per EM iteration with a record
        ``{"stage", "iter", "loglik", "theta", "c0", ...}``.
    keep_history
        Store the per-iteration EM trace in the result.

    Returns
    -------
    FitResult
    """
    t_start = time.perf_counter()
    spec = ModelSpec(n_components, stratification)
    cfg = em if em is not None else EMConfig(c0=c0, c0_max=max(20, c0), legacy=legacy)
    legacy = cfg.legacy
    double_count = legacy if sandwich_double_count is None else bool(sandwich_double_count)
    if (population_prevalence is None) != (sample_prevalence is None):
        raise InputError("Give both population_prevalence and sample_prevalence, or neither.")
    plan = plan or resolve_plan(device, threads=threads, chunk_size=chunk_size)
    nbh = neighborhoods if neighborhoods is not None else data.neighborhoods
    explicit, pics = _as_start_list(spec, starting, starting_pic)
    notes: list[str] = []
    timings: dict[str, float] = {}

    def make_callback(stage: str) -> Callable[[dict[str, Any]], None] | None:
        if progress is None:
            return None

        def cb(rec: dict[str, Any]) -> None:
            progress({"stage": stage, **rec})

        return cb

    with Engine(data, plan) as engine:
        log.info("Compute: %s; %s SNPs", engine.describe(), f"{data.K:,}")
        lik = CompositeLikelihood(spec, engine, cfg.c0)

        do_polish = (not legacy) if polish is None else bool(polish)

        # ------------------------------------------------------------ starting values
        starts: list[np.ndarray] = []
        prefits: list[dict[str, Any]] = []
        if explicit is not None:
            starts = explicit
        else:
            spec2 = ModelSpec(2, stratification)
            lik2 = lik if spec.n_components == 2 else CompositeLikelihood(spec2, engine, cfg.c0)
            for pic in pics:
                s2 = _two_component_start(lik2, data, pic, legacy, notes)
                if spec.n_components == 2:
                    starts.append(s2)
                else:
                    t0 = time.perf_counter()
                    pre, pre_pol = _em_then_newton(
                        lik2,
                        s2,
                        cfg.loosened(),
                        make_callback("prefit-M2"),
                        polish=do_polish,
                        accelerate=accelerate,
                        warmup=newton_after,
                    )
                    pre_theta = (
                        pre_pol.theta
                        if pre_pol is not None and pre_pol.status == "converged"
                        else pre.theta
                    )
                    timings["prefit_seconds"] = timings.get("prefit_seconds", 0.0) + (
                        time.perf_counter() - t0
                    )
                    prefits.append(
                        {
                            "starting": s2,
                            "estimates": pre_theta,
                            "loglik": pre.loglik,
                            "n_iter": pre.n_iter,
                            "status": pre.status,
                            "c0": pre.c0,
                        }
                    )
                    starts.append(m3_start_from_m2(pre_theta))

        # ------------------------------------------------------------- EM and Newton
        t0 = time.perf_counter()

        def final_loglik(em_r: EMResult, pol_r: PolishResult | None) -> float:
            if pol_r is not None and pol_r.n_steps > 0 and pol_r.status != "not_positive_definite":
                return pol_r.loglik
            return em_r.loglik

        def solve(
            start_list: list[np.ndarray], cfg_run: EMConfig, first_round: bool
        ) -> list[tuple[np.ndarray, EMResult, PolishResult | None]]:
            # A re-solve at a higher truncation level starts from the previous optimum,
            # so Newton's method is tried after a single EM iteration.
            warmup = newton_after if first_round else 1
            out: list[tuple[np.ndarray, EMResult, PolishResult | None]] = []
            for i, s in enumerate(start_list):
                stage = spec.name
                if len(start_list) > 1:
                    stage = f"{spec.name}[start {i + 1}/{len(start_list)}]"
                elif not first_round:
                    stage = f"{spec.name}[c0={cfg_run.c0}]"
                try:
                    em_i, pol_i = _em_then_newton(
                        lik,
                        s,
                        cfg_run,
                        make_callback(stage),
                        polish=do_polish,
                        accelerate=accelerate,
                        warmup=warmup,
                    )
                    out.append((s, em_i, pol_i))
                except GenesisError as exc:
                    if len(start_list) == 1:
                        raise
                    notes.append(f"Start {i + 1} failed: {exc}")
            if not out:
                raise GenesisError("All starting values failed; see the notes above.")
            return out

        # Truncation rule (default mode): solve at c0; if the prior mass discarded by the
        # truncation, evaluated at the solution, exceeds the tolerance, raise c0 and solve
        # again from that solution. Unlike the original's in-EM rule this depends only on
        # the data, not on the path from the starting values.
        c0_now = cfg.c0
        c0_path: list[dict[str, Any]] = []
        current_starts = starts
        first_runs: list[tuple[np.ndarray, EMResult, PolishResult | None]] | None = None
        first_best = 0
        full_history: list[dict[str, Any]] = []
        total_iter = 0
        total_backtracks = 0
        while True:
            cfg_run = cfg if legacy else replace(cfg, c0=c0_now)
            runs = solve(current_starts, cfg_run, first_runs is None)
            best_i = int(np.nanargmax([final_loglik(e, q) for _, e, q in runs]))
            _, emres, pol = runs[best_i]
            if first_runs is None:
                first_runs, first_best = runs, best_i
            for rec in emres.history:  # one continuous trace across truncation levels
                rec["iter"] += total_iter
            full_history.extend(emres.history)
            total_iter += emres.n_iter
            total_backtracks += emres.n_backtracks
            lik.set_c0(emres.c0)
            theta = emres.theta.copy()
            loglik, loglik_unnorm = emres.loglik, emres.loglik_unnorm
            if (
                pol is not None
                and pol.status in ("converged", "max_steps", "no_ascent")
                and pol.n_steps > 0
            ):
                theta, loglik, loglik_unnorm = pol.theta.copy(), pol.loglik, pol.loglik_unnorm
            if legacy:
                break
            tail = lik.truncation(theta).mean_tail
            c0_path.append(
                {"c0": c0_now, "mean_tail": tail, "estimates": theta.copy(), "loglik": loglik}
            )
            if tail <= cfg.tail_tol or c0_now >= cfg.c0_max:
                break
            if _degeneracy(spec, theta) is not None:
                break  # a model that is not identified is not worth refining further
            for cand in range(c0_now + 1, cfg.c0_max + 1):
                lik.set_c0(cand)
                c0_now = cand
                if lik.truncation(theta).mean_tail <= 0.1 * cfg.tail_tol:
                    break
            log.info(
                "Truncated prior mass %.2e exceeds %.0e at c0 = %d: re-solving with c0 = %d",
                tail,
                cfg.tail_tol,
                c0_path[-1]["c0"],
                c0_now,
            )
            current_starts = [theta.copy()]
        timings["em_seconds"] = time.perf_counter() - t0
        assert first_runs is not None
        start_used = first_runs[first_best][0]
        if (
            not legacy
            and c0_path[-1]["mean_tail"] > cfg.tail_tol
            and _degeneracy(spec, theta) is None
        ):
            msg = (
                f"Truncation still matters at the ceiling c0 = {c0_now}: on average "
                f"{c0_path[-1]['mean_tail']:.1e} of the prior mass is discarded (tolerance "
                f"{cfg.tail_tol:.0e}). Estimates, pi_c in particular, may be limited by the "
                "truncation; check their sensitivity with a larger c0_max."
            )
            notes.append(msg)
            warnings.warn(msg, NumericalWarning, stacklevel=2)

        polish_info: dict[str, Any] | None = None
        if pol is not None:
            polish_info = {
                "status": pol.status,
                "n_steps": pol.n_steps,
                "last_step_naive_se": pol.last_step,
                "fixed": list(pol.fixed),
                "em_estimates": emres.theta.copy(),
                "em_loglik": emres.loglik,
                "loglik_gain": pol.loglik - emres.loglik,
                "accelerated": bool(accelerate),
            }
            if pol.status != "converged":
                msg = {
                    "not_positive_definite": "Newton polishing was skipped because the observed "
                    "information is not positive definite at the EM estimate; the EM estimate is "
                    "reported.",
                    "no_ascent": "Newton polishing stopped early (no step increased the "
                    "likelihood); the best point found is reported.",
                    "max_steps": "Newton polishing reached its step limit before converging.",
                }[pol.status]
                notes.append(msg)
        runs = first_runs
        best_i = first_best
        multistart = [
            {
                "starting": s,
                "estimates": q.theta if q is not None and q.status == "converged" else r.theta,
                "loglik": final_loglik(r, q),
                "n_iter": r.n_iter,
                "status": r.status,
                "c0": r.c0,
                "selected": i == best_i,
            }
            for i, (s, r, q) in enumerate(runs)
        ]

        # The estimate is converged if the EM met its criteria or, failing that, Newton
        # polishing reached a stationary point from where the EM stopped.
        polished = polish_info is not None and polish_info["status"] == "converged"
        final_converged = emres.converged or (polished and emres.status in ("max_iter", "newton"))
        final_status = "converged" if final_converged else emres.status
        if polished and not emres.converged and emres.status == "max_iter":
            notes.append(
                f"The EM used all {cfg.max_iter} iterations; Newton polishing then converged to a "
                "stationary point of the likelihood, which is the reported estimate."
            )
        if not final_converged and emres.status != "not_identified":
            msg = {
                "max_iter": f"EM reached the iteration limit ({cfg.max_iter}) without meeting the "
                "convergence criteria; consider raising max_iter.",
                "stalled": "EM stalled: no step that increases the composite likelihood could be "
                "found. The estimate is the last accepted iterate.",
                "legacy_break": "legacy mode: the original algorithm's early exit was triggered "
                "(log-likelihood decreased at c0 >= 20). Reported parameters are those of the "
                "previous iteration, as in the original.",
            }.get(emres.status, f"EM ended with status {emres.status!r}.")
            notes.append(msg)
            warnings.warn(msg, ConvergenceWarning, stacklevel=2)

        # -------------------------------------------------------------- variance step
        cov = info = J = None
        se = np.full(spec.n_params, np.nan)
        ds = aic = bic = float("nan")
        diagnostics: dict[str, Any] = {
            "polish": polish_info,
            "em_status": emres.status,
            "c0_path": c0_path,
            "truncation": lik.truncation(theta).to_dict(),
            "ldsc": ldsc_regression(data).to_dict(),
            "multistart": multistart,
            "em_backtracks": total_backtracks,
        }
        if prefits:
            diagnostics["prefit_m2"] = prefits
        labels_swapped = False
        why = _degeneracy(spec, theta)
        degenerate = why is not None
        if degenerate:
            msg = (
                f"The three-component model is not identified for these data: {why}. "
                "p1 and the cluster-specific quantities are not meaningful and the standard "
                "errors are unreliable; use the two-component model."
            )
            notes.append(msg)
            warnings.warn(msg, NumericalWarning, stacklevel=2)
        diagnostics["degenerate"] = degenerate
        if variance:
            t0 = time.perf_counter()
            si = lik.score_info(theta)
            info_used = si.information
            if legacy and spec.n_components == 3:
                # The original Ik3() never fills rr(2,0) ("rr(2,0) = rr(2,0)"), so that
                # entry of its information matrix lacks the E[r r'] term.
                info_used = si.information.copy()
                info_used[2, 0] += si.e_rr[2, 0]
            free = np.ones(spec.n_params, dtype=bool) if legacy else spec.free.copy()
            bound_names: list[str] = []
            if not legacy:
                # Parameters on a bound are not free: the remaining standard errors are
                # conditional on the constraint and theirs are reported as NaN.
                if spec.n_components == 3:
                    for i, bounds in ((0, cfg.pic_bounds), (1, cfg.p1_bounds)):
                        if np.isclose(theta[i], bounds[0], rtol=1e-9) or np.isclose(
                            theta[i], bounds[1], rtol=1e-9
                        ):
                            free[i] = False
                            bound_names.append(spec.param_names[i])
                for i in spec.idx_sig:
                    if theta[i] <= 1e-11:
                        free[i] = False
                        bound_names.append(spec.param_names[i])
            if nbh is None:
                msg = (
                    "No tagging neighbourhoods were supplied: the sandwich estimator treats "
                    "markers as independent, which understates standard errors for real GWAS "
                    "data. Pass `neighborhoods=` (or prepare the data with an LD reference)."
                )
                notes.append(msg)
                warnings.warn(msg, DataWarning, stacklevel=2)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", NumericalWarning)
                sw = sandwich(
                    si.scores,
                    info_used,
                    nbh,
                    data.ref_index,
                    free=free,
                    self_double_count=double_count,
                )
            for w in caught:
                notes.append(str(w.message))
                warnings.warn(w.message, w.category, stacklevel=2)
            cov, info, J, ds = sw.covariance, sw.information, sw.J, sw.ds
            se = sw.se
            for name in bound_names:
                se[spec.param_names.index(name)] = np.nan
            if degenerate:
                # d_s = tr(I^-1 J) is meaningless when the information is singular.
                ds = float("nan")
            else:
                ic = information_criteria(loglik, ds, float(np.mean(data.n)), bic_gamma)
                aic, bic = ic["aic"], ic["bic"]

            # The other sandwich convention, so that both sets of standard errors are
            # available from one fit: J_original = J_paper + sum_k S_k S_k'.
            fidx = np.flatnonzero(free)
            outer = (si.scores.T @ si.scores)[np.ix_(fidx, fidx)]
            j_alt = J[np.ix_(fidx, fidx)] + (-outer if double_count else outer)
            inv_f = sw.inv_information[np.ix_(fidx, fidx)]
            cov_alt = inv_f @ j_alt @ inv_f
            se_alt = np.full(spec.n_params, np.nan)
            with np.errstate(invalid="ignore"):
                se_alt[fidx] = np.sqrt(np.where(np.diag(cov_alt) >= 0, np.diag(cov_alt), np.nan))
            ds_alt = float(np.trace(inv_f @ j_alt))
            diagnostics["sandwich_alt"] = {
                "convention": "paper" if double_count else "original_double_count",
                "se": se_alt,
                "ds": ds_alt,
                "bic": information_criteria(loglik, ds_alt, float(np.mean(data.n)), bic_gamma)[
                    "bic"
                ],
            }

            # Remaining distance to the stationary point of the EM objective, estimated
            # by one Newton step and expressed in standard-error units.
            idx = np.flatnonzero(free & spec.free)
            inv_sym, _ = scaled_inverse(si.information[np.ix_(idx, idx)])
            step = np.zeros(spec.n_params)
            step[idx] = inv_sym @ si.scores.sum(axis=0)[idx]
            with np.errstate(divide="ignore", invalid="ignore"):
                dist = np.where(se > 0, step / se, np.nan)
            diagnostics["newton_step"] = step
            diagnostics["newton_distance_se"] = dist
            diagnostics["information_pd"] = sw.information_pd
            diagnostics["information_condition"] = sw.condition_number
            finite = np.abs(dist[np.isfinite(dist)])
            if finite.size and finite.max() > 0.25 and not legacy and not degenerate:
                msg = (
                    f"The estimate is about {finite.max():.2f} standard errors from the "
                    "likelihood stationary point: the EM stopping rule (that of the original "
                    "software) is loose and Newton polishing was disabled or did not converge. "
                    "Enable polishing or tighten the tolerances (EMConfig / --tol-scale)."
                )
                notes.append(msg)
                warnings.warn(msg, ConvergenceWarning, stacklevel=2)
            timings["variance_seconds"] = time.perf_counter() - t0

        if spec.n_components == 3 and not legacy:
            theta, (cov, info, J), labels_swapped = canonicalize_m3(theta, cov, info, J)
            if labels_swapped:
                se = np.array([se[0], se[1], se[3], se[2], se[4]])
                notes.append(
                    "Component labels were swapped so that cluster 1 is the larger-variance "
                    "cluster (the fit ended with sigmasq1 < sigmasq2)."
                )
        diagnostics["labels_swapped"] = labels_swapped

        at_bound = []
        if spec.n_components == 3:
            bounds_by_name = {"pic": cfg.pic_bounds, "p1": cfg.p1_bounds}
            if labels_swapped:  # p1 was mapped to 1 - p1
                bounds_by_name["p1"] = (1.0 - cfg.p1_bounds[1], 1.0 - cfg.p1_bounds[0])
            for name, (lo, hi) in bounds_by_name.items():
                val = float(theta[spec.param_names.index(name)])
                if np.isclose(val, lo, rtol=1e-9) or np.isclose(val, hi, rtol=1e-9):
                    at_bound.append(name)
        if at_bound and not degenerate:
            msg = (
                f"Parameter(s) {', '.join(at_bound)} ended on a bound; the three-component model "
                "may be over-parameterised for these data (compare with M2 by BIC). Standard "
                "errors of the other parameters are conditional on the constraint."
            )
            notes.append(msg)
        diagnostics["at_bound"] = at_bound

        derived = derived_quantities(spec, theta, cov, data.M, legacy=legacy)
        if population_prevalence is not None and sample_prevalence is not None:
            from genesis_gwas.analysis.heritability import h2_transfer

            h2 = derived["heritability"]
            diagnostics["liability"] = h2_transfer(
                h2.estimate, h2.se, population_prevalence, sample_prevalence
            )
        engine_desc = engine.describe()

    timings["total_seconds"] = time.perf_counter() - t_start
    settings = {
        "em": asdict(cfg),
        "legacy": legacy,
        "sandwich_double_count": double_count,
        "bic_gamma": bic_gamma,
        "variance": variance,
        "polish": do_polish,
        "accelerate": bool(accelerate and do_polish),
        "newton_after": newton_after,
        "has_neighborhoods": nbh is not None,
    }
    provenance = {
        "software": "genesis-gwas",
        "version": __version__,
        "compute": {**plan.to_dict(), "description": engine_desc},
        "data": dict(data.meta),
    }
    return FitResult(
        n_components=spec.n_components,
        stratification=stratification,
        param_names=spec.param_names,
        estimates=theta,
        se=se,
        covariance=cov,
        loglik=loglik,
        loglik_unnorm=loglik_unnorm,
        c0=emres.c0,
        n_iter=total_iter,
        converged=final_converged,
        status=final_status,
        n_snps=data.K,
        M=data.M,
        mean_n=float(np.mean(data.n)),
        information=info,
        J=J,
        ds=ds,
        aic=aic,
        bic=bic,
        bic_gamma=bic_gamma,
        derived=derived,
        starting=np.asarray(start_used, dtype=float),
        diagnostics=diagnostics,
        history=full_history if keep_history else [],
        settings=settings,
        provenance=provenance,
        timings=timings,
        warnings=notes,
    )
