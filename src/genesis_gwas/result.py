"""Result container for a fitted model."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from genesis_gwas.core.derived import DERIVED_LABELS, Derived
from genesis_gwas.core.spec import ModelSpec
from genesis_gwas.exceptions import InputError

__all__ = ["FitResult", "ModelComparison", "select_model"]

RESULT_SCHEMA = 1


def _jsonable(x: Any) -> Any:
    """Convert numpy containers to plain Python; non-finite floats become ``None``."""
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return v if math.isfinite(v) else None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, Path):
        return str(x)
    return x


def _arr(x: Any) -> np.ndarray | None:
    if x is None:
        return None
    return np.array([[np.nan if v is None else v for v in row] for row in x], dtype=float) if (
        len(x) and isinstance(x[0], list)
    ) else np.array([np.nan if v is None else v for v in x], dtype=float)


def _num(x: Any) -> float:
    return float("nan") if x is None else float(x)


@dataclass
class FitResult:
    """Everything produced by :func:`genesis_gwas.fit`.

    Attributes
    ----------
    n_components
        ``2`` or ``3``.
    stratification
        Whether the variance-inflation parameter ``a`` was estimated.
    param_names
        Names of the parameters, in the order of ``estimates``.
    estimates, se
        Parameter estimates and sandwich standard errors.
    covariance
        Sandwich covariance matrix of ``estimates`` (``None`` if not computed).
    loglik
        Renormalised truncated composite log-likelihood at the estimate (the value the
        original software reports).
    loglik_unnorm
        Unnormalised truncated composite log-likelihood (the EM objective).
    c0
        Truncation level in force at the end of the fit.
    n_iter, converged, status
        EM bookkeeping; ``status`` is ``"converged"``, ``"max_iter"``, ``"stalled"`` or
        ``"legacy_break"``.
    n_snps, M, mean_n
        Number of GWAS markers analysed, number of reference SNPs, mean sample size.
    information, J, ds
        Observed information, score-variability matrix and the effective number of
        parameters ``tr(I^-1 J)``.
    aic, bic, bic_gamma
        Composite-likelihood information criteria (lower is better).
    derived
        Interpretable quantities (number of susceptibility SNPs, heritability, ...)
        with delta-method standard errors; see :mod:`genesis_gwas.core.derived`.
    starting
        Starting values used for the selected EM run.
    diagnostics
        Truncation diagnostics, the remaining Newton distance to the stationary point
        (in standard-error units), LD-score-regression summary, multi-start table, ...
    history
        Per-iteration trace of the selected EM run (iteration, log-likelihoods,
        parameters, ``c0``).
    settings, provenance, timings
        Algorithm settings, software/hardware provenance and wall-clock timings.
    warnings
        Messages the analyst should read (non-convergence, parameters on bounds, ...).
    """

    n_components: int
    stratification: bool
    param_names: tuple[str, ...]
    estimates: np.ndarray
    se: np.ndarray
    covariance: np.ndarray | None
    loglik: float
    loglik_unnorm: float
    c0: int
    n_iter: int
    converged: bool
    status: str
    n_snps: int
    M: int
    mean_n: float
    information: np.ndarray | None = None
    J: np.ndarray | None = None
    ds: float = float("nan")
    aic: float = float("nan")
    bic: float = float("nan")
    bic_gamma: float = 0.5
    derived: dict[str, Derived] = field(default_factory=dict)
    starting: np.ndarray | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    settings: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    # ------------------------------------------------------------- conveniences
    @property
    def spec(self) -> ModelSpec:
        """The :class:`~genesis_gwas.core.spec.ModelSpec` of this fit."""
        return ModelSpec(self.n_components, self.stratification)

    @property
    def model(self) -> str:
        """``"M2"`` or ``"M3"``."""
        return f"M{self.n_components}"

    @property
    def params(self) -> dict[str, float]:
        """``{parameter name: estimate}``."""
        return {k: float(v) for k, v in zip(self.param_names, self.estimates)}

    @property
    def pic(self) -> float:
        """Estimated proportion of susceptibility SNPs."""
        return float(self.estimates[0])

    @property
    def a(self) -> float:
        """Estimated variance-inflation parameter."""
        return float(self.estimates[-1])

    @property
    def mixture(self) -> tuple[np.ndarray, np.ndarray]:
        """Weights and variances of the effect-size distribution of susceptibility SNPs.

        Returns ``(p, sigmasq)`` with ``p`` summing to one — ``([1.0], [sigmasq])`` for
        M2 and ``([p1, 1 - p1], [sigmasq1, sigmasq2])`` for M3.
        """
        _, p, sig, _ = self.spec.unpack(self.estimates)
        return p, sig

    @property
    def heritability(self) -> float:
        """Estimated total heritability (observed or log-odds-ratio scale)."""
        return self.derived["heritability"].estimate

    @property
    def n_ssnp(self) -> float:
        """Estimated number of susceptibility SNPs."""
        return self.derived["n_ssnp"].estimate

    # ------------------------------------------------------------ serialisation
    def to_dict(self, *, include_history: bool = True) -> dict[str, Any]:
        """Plain-Python (JSON-serialisable) representation. Non-finite numbers become ``None``."""
        d: dict[str, Any] = {
            "schema": RESULT_SCHEMA,
            "model": self.model,
            "n_components": self.n_components,
            "stratification": self.stratification,
            "param_names": list(self.param_names),
            "estimates": self.estimates,
            "se": self.se,
            "covariance": self.covariance,
            "loglik": self.loglik,
            "loglik_unnorm": self.loglik_unnorm,
            "c0": self.c0,
            "n_iter": self.n_iter,
            "converged": self.converged,
            "status": self.status,
            "n_snps": self.n_snps,
            "M": self.M,
            "mean_n": self.mean_n,
            "information": self.information,
            "J": self.J,
            "ds": self.ds,
            "aic": self.aic,
            "bic": self.bic,
            "bic_gamma": self.bic_gamma,
            "derived": {k: v.to_dict() for k, v in self.derived.items()},
            "starting": self.starting,
            "diagnostics": self.diagnostics,
            "settings": self.settings,
            "provenance": self.provenance,
            "timings": self.timings,
            "warnings": list(self.warnings),
        }
        if include_history:
            d["history"] = self.history
        return _jsonable(d)

    def to_json(self, path: str | Path | None = None, *, include_history: bool = True) -> str:
        """Serialise to JSON; if ``path`` is given the text is also written there."""
        text = json.dumps(self.to_dict(include_history=include_history), indent=2, allow_nan=False)
        if path is not None:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_name(p.name + ".tmp")
            tmp.write_text(text + "\n")
            tmp.replace(p)
        return text

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FitResult:
        """Inverse of :meth:`to_dict`."""
        if d.get("schema", RESULT_SCHEMA) > RESULT_SCHEMA:
            raise InputError(
                f"Result file uses schema {d.get('schema')}, newer than this version of "
                f"genesis-gwas understands ({RESULT_SCHEMA}). Please upgrade."
            )
        history = []
        for rec in d.get("history", []) or []:
            rec = dict(rec)
            if "theta" in rec:
                rec["theta"] = _arr(rec["theta"])
            history.append(rec)
        est = _arr(d["estimates"])
        se = _arr(d["se"])
        assert est is not None and se is not None
        return cls(
            n_components=int(d["n_components"]),
            stratification=bool(d["stratification"]),
            param_names=tuple(d["param_names"]),
            estimates=est,
            se=se,
            covariance=_arr(d.get("covariance")),
            loglik=_num(d["loglik"]),
            loglik_unnorm=_num(d.get("loglik_unnorm")),
            c0=int(d["c0"]),
            n_iter=int(d["n_iter"]),
            converged=bool(d["converged"]),
            status=str(d["status"]),
            n_snps=int(d["n_snps"]),
            M=int(d["M"]),
            mean_n=_num(d["mean_n"]),
            information=_arr(d.get("information")),
            J=_arr(d.get("J")),
            ds=_num(d.get("ds")),
            aic=_num(d.get("aic")),
            bic=_num(d.get("bic")),
            bic_gamma=_num(d.get("bic_gamma", 0.5)),
            derived={
                k: Derived(_num(v["estimate"]), _num(v["se"])) for k, v in (d.get("derived") or {}).items()
            },
            starting=_arr(d.get("starting")),
            diagnostics=d.get("diagnostics") or {},
            history=history,
            settings=d.get("settings") or {},
            provenance=d.get("provenance") or {},
            timings=d.get("timings") or {},
            warnings=list(d.get("warnings") or []),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> FitResult:
        """Load a result written by :meth:`to_json`."""
        return cls.from_dict(json.loads(Path(path).read_text()))

    # ------------------------------------------------------------------ reports
    def summary(self) -> str:
        """Multi-line human-readable report."""
        kind = "two-component" if self.n_components == 2 else "three-component"
        lines = [
            f"GENESIS fit: {self.model} ({kind} model)"
            + ("" if self.stratification else "  [stratification term a fixed at 0]"),
            f"  GWAS markers analysed (K)     : {self.n_snps:,}",
            f"  Reference-panel SNPs (M)      : {self.M:,}",
            f"  Mean sample size              : {self.mean_n:,.1f}",
            f"  EM                            : {self.status} after {self.n_iter:,} iteration(s), c0 = {self.c0}"
            + self._polish_note(),
            f"  Composite log-likelihood      : {self.loglik:,.4f}",
        ]
        if math.isfinite(self.bic):
            lines.append(
                f"  Model selection               : BIC = {self.bic:,.3f}   AIC = {self.aic:,.3f}   "
                f"d_s = {self.ds:.3f}"
            )
        lines += ["", f"  {'Parameter':<12}{'Estimate':>16}{'SE':>16}"]
        for name, est, se in zip(self.param_names, self.estimates, self.se):
            lines.append(f"  {name:<12}{est:>16.6e}{se:>16.4e}")
        if self.derived:
            lines += ["", f"  {'Derived quantity':<56}{'Estimate':>14}{'SE':>12}"]
            for key, val in self.derived.items():
                label = DERIVED_LABELS.get(key, key)
                lines.append(f"  {label:<56}{val.estimate:>14.5g}{val.se:>12.4g}")
        liab = self.diagnostics.get("liability")
        if liab:
            lines += [
                "",
                "  Heritability on other scales (binary trait)",
                f"    observed scale   : {liab['h2_observed']:.5g} (SE {liab['se_h2_observed']:.4g})",
                f"    liability scale  : {liab['h2_liability']:.5g} (SE {liab['se_h2_liability']:.4g})",
            ]
        dist = self.diagnostics.get("newton_distance_se")
        if dist is not None:
            finite = [abs(v) for v in dist if v is not None and math.isfinite(v)]
            if finite:
                lines += [
                    "",
                    f"  Convergence check             : estimate is within {max(finite):.3g} SE "
                    "of the likelihood stationary point",
                ]
        if self.warnings:
            lines += ["", "  Warnings:"]
            lines += [f"    - {w}" for w in self.warnings]
        return "\n".join(lines)

    def _polish_note(self) -> str:
        pol = self.diagnostics.get("polish")
        if not pol:
            return ""
        if pol.get("status") == "converged":
            return f"; {pol.get('n_steps', 0)} Newton polishing step(s)"
        return f"; Newton polishing: {pol.get('status')}"

    def __str__(self) -> str:
        return self.summary()


@dataclass(frozen=True)
class ModelComparison:
    """Outcome of :func:`select_model`.

    Attributes
    ----------
    best
        The fit with the smallest criterion value.
    criterion
        ``"bic"`` or ``"aic"``.
    table
        One row per candidate: model label, log-likelihood, ``ds``, AIC, BIC and the
        difference in the criterion relative to the best model.
    """

    best: FitResult
    criterion: str
    table: list[dict[str, Any]]

    def summary(self) -> str:
        """Small text table of the comparison."""
        lines = [
            f"Model comparison by {self.criterion.upper()} (lower is better)",
            f"  {'Model':<8}{'logLik':>18}{'d_s':>10}{'AIC':>18}{'BIC':>18}{'delta':>12}",
        ]
        for row in self.table:
            mark = "  <- selected" if row["selected"] else ("  (not identified)" if row.get("degenerate") else "")
            lines.append(
                f"  {row['model']:<8}{row['loglik']:>18,.3f}{row['ds']:>10.3f}{row['aic']:>18,.3f}"
                f"{row['bic']:>18,.3f}{row['delta']:>12.3f}{mark}"
            )
        return "\n".join(lines)


def select_model(fits: list[FitResult], criterion: str = "bic") -> ModelComparison:
    """Choose between fitted models with the composite-likelihood BIC (default) or AIC.

    All candidates must have been fitted to the same data. A fit flagged as degenerate
    (a three-component fit whose variance components merged or collapsed, i.e. a model
    that is not identified for the data) is never selected when a non-degenerate
    candidate exists.
    """
    if criterion not in ("bic", "aic"):
        raise InputError("criterion must be 'bic' or 'aic'")
    if not fits:
        raise InputError("No fits to compare")
    if len({f.n_snps for f in fits}) != 1:
        raise InputError("Models were fitted to different numbers of SNPs and cannot be compared")
    vals = [getattr(f, criterion) for f in fits]
    if not all(math.isfinite(v) for v in vals):
        raise InputError(
            f"{criterion.upper()} is unavailable for at least one fit (was the variance step skipped?)"
        )
    # A degenerate fit (e.g. M3 whose two variance components merged) is not a valid
    # candidate when a well-identified alternative exists.
    eligible = [i for i, f in enumerate(fits) if not f.diagnostics.get("degenerate", False)]
    pool = eligible if eligible else list(range(len(fits)))
    best_i = min(pool, key=lambda i: vals[i])
    table = [
        {
            "model": f.model,
            "loglik": f.loglik,
            "ds": f.ds,
            "aic": f.aic,
            "bic": f.bic,
            "delta": v - vals[best_i],
            "selected": i == best_i,
            "degenerate": bool(f.diagnostics.get("degenerate", False)),
        }
        for i, (f, v) in enumerate(zip(fits, vals))
    ]
    return ModelComparison(best=fits[best_i], criterion=criterion, table=table)
