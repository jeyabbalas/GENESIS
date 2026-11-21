"""High-level model fitting: starting values → EM → sandwich variance → derived quantities."""

from __future__ import annotations

import logging
import time
import warnings
from collections.abc import Callable, Sequence
from dataclasses import asdict
from typing import Any

import numpy as np

from genesis_gwas._version import __version__
from genesis_gwas.compute import ComputePlan, resolve_plan
from genesis_gwas.core.data import ModelData
from genesis_gwas.core.derived import canonicalize_m3, derived_quantities
from genesis_gwas.core.em import EMConfig, EMResult, run_em
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
                    pre = run_em(lik2, s2, cfg.loosened(), make_callback("prefit-M2"))
                    timings["prefit_seconds"] = timings.get("prefit_seconds", 0.0) + (
                        time.perf_counter() - t0
                    )
                    prefits.append(
                        {
                            "starting": s2,
                            "estimates": pre.theta,
                            "loglik": pre.loglik,
                            "n_iter": pre.n_iter,
                            "status": pre.status,
                            "c0": pre.c0,
                        }
                    )
                    starts.append(m3_start_from_m2(pre.theta))

        # ------------------------------------------------------------------------ EM
        t0 = time.perf_counter()
        runs: list[tuple[np.ndarray, EMResult]] = []
        for i, s in enumerate(starts):
            stage = spec.name if len(starts) == 1 else f"{spec.name}[start {i + 1}/{len(starts)}]"
            try:
                runs.append((s, run_em(lik, s, cfg, make_callback(stage))))
            except GenesisError as exc:
                if len(starts) == 1:
                    raise
                notes.append(f"Start {i + 1} failed: {exc}")
        if not runs:
            raise GenesisError("All starting values failed; see the notes above.")
        timings["em_seconds"] = time.perf_counter() - t0
        best_i = int(np.nanargmax([r.loglik for _, r in runs]))
        start_used, emres = runs[best_i]
        lik.set_c0(emres.c0)
        theta = emres.theta.copy()
        multistart = [
            {
                "starting": s,
                "estimates": r.theta,
                "loglik": r.loglik,
                "n_iter": r.n_iter,
                "status": r.status,
                "c0": r.c0,
                "selected": i == best_i,
            }
            for i, (s, r) in enumerate(runs)
        ]

        if not emres.converged:
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
            "truncation": lik.truncation(theta).to_dict(),
            "ldsc": ldsc_regression(data).to_dict(),
            "multistart": multistart,
            "em_backtracks": emres.n_backtracks,
        }
        if prefits:
            diagnostics["prefit_m2"] = prefits
        labels_swapped = False
        if variance:
            t0 = time.perf_counter()
            si = lik.score_info(theta)
            info_used = si.information
            if legacy and spec.n_components == 3:
                # The original Ik3() never fills rr(2,0) ("rr(2,0) = rr(2,0)"), so that
                # entry of its information matrix lacks the E[r r'] term.
                info_used = si.information.copy()
                info_used[2, 0] += si.e_rr[2, 0]
            free = np.ones(spec.n_params, dtype=bool) if legacy else spec.free
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
            ic = information_criteria(emres.loglik, ds, float(np.mean(data.n)), bic_gamma)
            aic, bic = ic["aic"], ic["bic"]

            # Remaining distance to the stationary point of the EM objective, estimated
            # by one Newton step and expressed in standard-error units.
            idx = np.flatnonzero(spec.free)
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
            if finite.size and finite.max() > 0.25 and not legacy:
                msg = (
                    f"The EM stopped about {finite.max():.2f} standard errors from the "
                    "likelihood stationary point (the stopping rule of the original software is "
                    "loose). Tighten the tolerances (EMConfig / --tol-scale) for a fully "
                    "converged estimate."
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
            pic_hat, p1_hat = float(emres.theta[0]), float(emres.theta[1])
            if np.isclose(pic_hat, cfg.pic_bounds[0]) or np.isclose(pic_hat, cfg.pic_bounds[1]):
                at_bound.append("pic")
            if np.isclose(p1_hat, cfg.p1_bounds[0]) or np.isclose(p1_hat, cfg.p1_bounds[1]):
                at_bound.append("p1")
        if at_bound:
            msg = (
                f"Parameter(s) {', '.join(at_bound)} ended on a bound; the three-component model "
                "may be over-parameterised for these data (compare with M2 by BIC) and the "
                "standard errors do not account for the constraint."
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
        loglik=emres.loglik,
        loglik_unnorm=emres.loglik_unnorm,
        c0=emres.c0,
        n_iter=emres.n_iter,
        converged=emres.converged,
        status=emres.status,
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
        history=emres.history if keep_history else [],
        settings=settings,
        provenance=provenance,
        timings=timings,
        warnings=notes,
    )
