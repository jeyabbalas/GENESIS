r"""Simulation from the fitted model: synthetic summary statistics, QQ envelopes, future λ.

Three generators of increasing realism are provided.

:func:`simulate_sumstats`
    Draws each marker independently from the **composite model** itself (eq. 3 of Zhang
    et al. 2018): latent counts of tagged susceptibility SNPs, then
    :math:`\hat\beta_k\sim N(0,\ \tfrac{\ell_k}{N^*_k}\sum_h J_h\sigma_h^2 + s_k^2 + a)`.
    This is the distribution the estimator assumes marker by marker, so it is the natural
    tool for checking parameter recovery.
:func:`simulate_ld_sumstats`
    The paper's "Simulation studies" scheme on a real LD reference: joint effects for
    every reference SNP from eq. (1), marginal effects through the tagging
    neighbourhoods, and LD-correlated sampling noise.
:func:`qq_simulation` and :func:`future_gc`
    Ports of ``qqplotdata.simu()`` and ``futuregc()`` of the original package, built on
    the same ingredients.

All random numbers come from :func:`numpy.random.default_rng`; replicate ``i`` of a
simulation uses the independent stream ``default_rng([seed, i])``, so results do not
depend on how many replicates are requested. (The original uses R's generator with
``set.seed(seeds * i)``; individual draws therefore cannot be reproduced, only their
distribution.)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import log_ndtr

from genesis_gwas.analysis.density import EffectModel, resolve_model
from genesis_gwas.core.data import ModelData
from genesis_gwas.core.spec import config_table, log_multinomial_table
from genesis_gwas.data.reference import LDReference
from genesis_gwas.exceptions import DataWarning, InputError
from genesis_gwas.result import FitResult

__all__ = [
    "QQData",
    "future_gc",
    "lambda_gc",
    "qq_simulation",
    "simulate_ld_sumstats",
    "simulate_sumstats",
]

_CHI2_MEDIAN = 0.45493642311957283  # qchisq(0.5, 1)
_LN10 = float(np.log(10.0))
_LN2 = float(np.log(2.0))


# --------------------------------------------------------------------------- helpers
def lambda_gc(z: Any) -> float:
    r"""Genomic-control factor :math:`\lambda_{GC}=\operatorname{median}(z^2)/0.4549`."""
    zv = np.asarray(z, dtype=np.float64)
    return float(np.median(zv * zv) / _CHI2_MEDIAN)


def _neg_log10_p(z: np.ndarray) -> np.ndarray:
    """Two-sided ``-log10(p)`` computed in log space (finite for arbitrarily large |z|)."""
    nlp = -(_LN2 + log_ndtr(-np.abs(z))) / _LN10
    return np.clip(nlp, 0.0, None) + 0.0  # also turns -0.0 (z == 0) into +0.0


def _rng(seed: int | None, stream: int) -> np.random.Generator:
    if seed is None:
        return np.random.default_rng()
    return np.random.default_rng([int(seed), int(stream)])


def _thin_index(size: int, max_points: int) -> np.ndarray:
    """Return indices into an ascending array: the upper tail in full, the bulk thinned."""
    if max_points < 2:
        raise InputError("max_points must be >= 2")
    if size <= max_points:
        return np.arange(size, dtype=np.int64)
    n_tail = max_points // 2
    tail = np.arange(size - n_tail, size, dtype=np.int64)
    bulk = np.unique(
        np.round(np.linspace(0, size - n_tail - 1, max_points - n_tail)).astype(np.int64)
    )
    return np.concatenate([bulk, tail])


class _ConfigSampler:
    r"""Sampler of the latent variance multiplier :math:`c_k=\sum_h J_h\sigma_h^2` of each marker.

    ``c0=None`` draws the counts from the untruncated binomial / multinomial law.
    Otherwise counts are drawn from that law *conditioned on* every count being
    ``<= c0`` — the truncated mixture the likelihood uses — by inverting the tabulated
    distribution function of each distinct :math:`N^*` in one vectorised search.
    """

    def __init__(self, nstar: np.ndarray, model: EffectModel, c0: int | None) -> None:
        self.model = model
        self.nstar = np.asarray(nstar, dtype=np.int64)
        self.c0 = c0
        if c0 is None:
            return
        if int(c0) < 1:
            raise InputError("c0 must be >= 1")
        H = model.weights.size
        J = config_table(H, int(c0))
        uniq, self._uidx = np.unique(self.nstar, return_inverse=True)
        logc = log_multinomial_table(uniq, J)
        b, _ = model.spec.log_weights(model.theta, J)
        logp = logc + b[None, :]
        logp -= logp.max(axis=1, keepdims=True)
        prob = np.exp(logp)
        cdf = np.cumsum(prob, axis=1)
        cdf /= cdf[:, -1:]
        T = J.shape[0]
        last_valid = (T - 1) - np.argmax(prob[:, ::-1] > 0, axis=1)
        cdf[np.arange(T)[None, :] >= last_valid[:, None]] = 1.0
        self._T = T
        self._last_valid = last_valid
        self._flat = (cdf + np.arange(uniq.size, dtype=np.float64)[:, None]).ravel()
        self._multiplier = J.astype(np.float64) @ model.variances
        self._counts = J

    def draw(self, rng: np.random.Generator) -> np.ndarray:
        """One draw of the multiplier for every marker, shape ``(K,)``."""
        if self.c0 is None:
            return self.draw_counts(rng).astype(np.float64) @ self.model.variances
        return self._multiplier[self._draw_index(rng)]

    def draw_counts(self, rng: np.random.Generator) -> np.ndarray:
        """One draw of the latent counts, shape ``(K, H)``."""
        if self.c0 is not None:
            return self._counts[self._draw_index(rng)]
        m = self.model
        if m.weights.size == 1:
            return rng.binomial(self.nstar, m.pic)[:, None]
        pi1 = m.pic * m.weights[0]
        pi2 = m.pic * m.weights[1]
        j1 = rng.binomial(self.nstar, pi1)
        j2 = rng.binomial(self.nstar - j1, min(1.0, pi2 / (1.0 - pi1)) if pi1 < 1.0 else 0.0)
        return np.stack([j1, j2], axis=1)

    def _draw_index(self, rng: np.random.Generator) -> np.ndarray:
        q = self._uidx + rng.random(self.nstar.size)
        t = np.searchsorted(self._flat, q, side="right") - self._uidx * self._T
        return np.minimum(t, self._last_valid[self._uidx])


def _positive_inflation(a: float) -> float:
    """Return ``a`` if positive, else 0 (a negative variance cannot be simulated)."""
    return a if a > 0 else 0.0


def _draw_joint_effects(model: EffectModel, size: int, rng: np.random.Generator) -> np.ndarray:
    """Joint effects of ``size`` reference SNPs from the spike-and-slab mixture (eq. 1)."""
    u = rng.random(size)
    edges = model.pic * np.cumsum(model.weights)
    edges[-1] = model.pic
    comp = np.searchsorted(edges, u, side="right")  # == H for null SNPs
    beta = np.zeros(size)
    causal = comp < model.weights.size
    beta[causal] = rng.standard_normal(int(causal.sum())) * model.sds[comp[causal]]
    return beta


def _ld_marginal(
    reference: LDReference, beta_joint: np.ndarray, block: int = 1 << 16
) -> np.ndarray:
    r"""Marginal effects :math:`\beta_k=\sum_{m\in\mathcal N_k}\beta^{(J)}_m\,|r_{km}|`.

    ``|r|`` is the square root of the stored pairwise :math:`r^2` — the sign of the
    correlation is not available in the reference, exactly as in the original
    ``futuregc()``. Because joint effects are symmetric about zero and independent
    across SNPs, the distribution of each marginal effect is unaffected.
    """
    nb = reference.neighborhoods
    indptr, indices = nb.indptr, nb.indices
    r2 = reference.pair_r2()
    size = reference.M
    if beta_joint.shape != (size,):
        raise InputError("beta_joint must have one entry per reference SNP")
    out = np.zeros(size)
    for r0 in range(0, size, block):
        r1 = min(size, r0 + block)
        lo, hi = int(indptr[r0]), int(indptr[r1])
        if hi == lo:
            continue
        vals = beta_joint[indices[lo:hi]] * np.sqrt(r2[lo:hi].astype(np.float64))
        lens = np.diff(indptr[r0 : r1 + 1])
        nonempty = lens > 0
        offs = (indptr[r0:r1] - lo)[nonempty]
        seg = out[r0:r1]
        seg[nonempty] = np.add.reduceat(vals, offs)
    return out


def _null_source(
    reference: LDReference | None, want: int, what: str, need_rows: bool
) -> tuple[int, bool]:
    """Decide how many replicates can use LD-correlated null Z-scores from the reference."""
    available = 0 if reference is None else reference.n_null_replicates
    if available == 0 or not need_rows:
        warnings.warn(
            f"{what}: no LD-correlated null replicates are available "
            "(no reference, a reference without null Z-scores, or data without reference "
            "rows); using independent N(0, 1) errors. The simulated spread then ignores LD "
            "and is too narrow.",
            DataWarning,
            stacklevel=3,
        )
        return want, False
    if want > available:
        warnings.warn(
            f"{what}: {want} simulations requested but the reference provides only "
            f"{available} null replicates; using {available}.",
            DataWarning,
            stacklevel=3,
        )
        return available, True
    return want, True


def _select_rows(rows: Any, size: int) -> np.ndarray:
    """Validate a row selection (``None``, boolean mask or integer index) of the reference."""
    if rows is None:
        return np.arange(size, dtype=np.int64)
    raw = np.asarray(rows)
    if raw.dtype == bool:
        if raw.size != size:
            raise InputError("A boolean `rows` mask must have one entry per reference SNP")
        raw = np.flatnonzero(raw)
    sel: np.ndarray = raw.astype(np.int64).ravel()
    if (
        sel.size == 0
        or int(np.min(sel)) < 0
        or int(np.max(sel)) >= size
        or np.unique(sel).size != sel.size
    ):
        raise InputError("rows must be unique reference row indices")
    return sel


# ------------------------------------------------------------------ model-based draws
def simulate_sumstats(
    nstar: Any,
    ldscore: Any,
    n: Any,
    theta: Any,
    *,
    M: int | None = None,
    c0: int | None = None,
    seed: int | None = None,
    snp: Any = None,
) -> ModelData:
    r"""Simulate summary statistics marker by marker from the composite model.

    For every marker :math:`k` (independently):

    1. draw the numbers of tagged susceptibility SNPs
       :math:`(N_k^{(1)},\dots,N_k^{(H)})\sim\text{Multinomial}(N^*_k;\ \pi_c p_1,\dots,\pi_c p_H)`
       (binomial for the two-component model);
    2. draw :math:`\hat\beta_k \sim N\bigl(0,\ \tfrac{\ell_k}{N^*_k}\sum_h N_k^{(h)}\sigma_h^2
       + \tfrac1{n_k} + a\bigr)`.

    Parameters
    ----------
    nstar, ldscore
        :math:`N^*_k` and :math:`\ell_k` of the markers (e.g. from an
        :class:`~genesis_gwas.data.reference.LDReference`).
    n
        Sample size(s): scalar or one value per marker.
    theta
        Parameters ``(pic, sigmasq, a)`` or ``(pic, p1, sigmasq1, sigmasq2, a)``, or a
        :class:`~genesis_gwas.result.FitResult`.
    M
        Reference-panel size recorded in the returned data (default: that of the fit,
        else 1,070,777).
    c0
        If given, the counts are drawn conditional on each being ``<= c0`` — i.e. from
        the *truncated* model that the likelihood evaluates. ``None`` (default) draws
        from the untruncated model.
    seed
        Seed of the random-number generator.
    snp
        Optional marker identifiers.

    Returns
    -------
    ModelData
        Independent markers (no tagging neighbourhoods). ``.meta["simulated"]`` records
        the generating parameters.
    """
    model = resolve_model(theta, M)
    ns = np.asarray(nstar)
    ld = np.asarray(ldscore, dtype=np.float64).ravel()
    K = ld.size
    nv = np.asarray(n, dtype=np.float64)
    nv = np.full(K, float(nv)) if nv.ndim == 0 else nv.ravel()
    if ns.size != K or nv.size != K:
        raise InputError("nstar, ldscore and n must have the same length")
    if K == 0:
        raise InputError("Nothing to simulate: zero markers")
    if np.any(nv <= 0) or np.any(ns < 1) or np.any(ld <= 0):
        raise InputError("Need n > 0, nstar >= 1 and ldscore > 0")
    rng = _rng(seed, 0)
    sampler = _ConfigSampler(ns.astype(np.int64), model, c0)
    mult = sampler.draw(rng)
    var = mult * ld / ns + 1.0 / nv + model.a
    if np.any(var <= 0):
        raise InputError("a is too negative: a + 1/n must be positive for every marker")
    betahat = rng.standard_normal(K) * np.sqrt(var)
    meta = {
        "simulated": {
            "scheme": "composite-model",
            "theta": model.theta.tolist(),
            "c0": c0,
            "seed": seed,
        }
    }
    return ModelData(
        betahat=betahat,
        varbetahat=1.0 / nv,
        ldscore=ld,
        nstar=ns,
        n=nv,
        M=model.M,
        snp=None if snp is None else np.asarray(snp),
        meta=meta,
    )


def simulate_ld_sumstats(
    reference: LDReference,
    theta: Any,
    n: Any,
    *,
    seed: int | None = None,
    replicate: int | None = 0,
    rows: Any = None,
    M: int | None = None,
) -> ModelData:
    r"""Simulate a GWAS on a real LD reference (the paper's "Simulation studies" scheme).

    .. math::

        \hat\beta_k = \beta_k + \nu_k + e_k, \qquad
        \beta_k = \sum_{m\in\mathcal N_k}\beta^{(J)}_m\,|r_{km}|, \qquad
        \nu_k \overset{iid}{\sim} N(0, a), \qquad e = \tilde u/\sqrt n,

    where the joint effects :math:`\beta^{(J)}_m` of all :math:`M` reference SNPs are
    drawn from the spike-and-slab mixture (eq. 1), :math:`\mathcal N_k` is the tagging
    neighbourhood of marker :math:`k` in the reference's LD configuration, and
    :math:`\tilde u` is one of the reference's stored null Z-score replicates (a null
    GWAS on the reference genotypes, hence correlated like the LD matrix).

    The marginal effects use :math:`|r_{km}|=\sqrt{r^2_{km}}`, as ``futuregc()`` of the
    original does, because only :math:`r^2` is stored; pairs with :math:`r^2` below the
    reference's tagging threshold do not contribute.

    Parameters
    ----------
    reference
        LD reference with pairwise :math:`r^2` for the selected configuration.
    theta
        Generating parameters (vector or :class:`~genesis_gwas.result.FitResult`). A
        negative ``a`` is treated as zero.
    n
        GWAS sample size: scalar, or one value per reference SNP.
    seed
        Seed of the random-number generator.
    replicate
        Which stored null replicate supplies the estimation error (0-based); ``None``
        draws independent :math:`N(0,1)` errors instead.
    rows
        Reference rows to return as GWAS markers (default: all), e.g. to mimic a study
        that genotyped only part of the panel. Effects are always simulated for the
        whole panel.
    M
        Reference-panel size recorded in the returned data (default ``reference.M``).

    Returns
    -------
    ModelData
        Built as :func:`genesis_gwas.prepare` would: with ``ref_index``, marker ids and
        the reference's tagging neighbourhoods, ready for :func:`genesis_gwas.fit`.
    """
    model = resolve_model(theta, reference.M if M is None else M)
    size = reference.M
    nv = np.asarray(n, dtype=np.float64)
    nv = np.full(size, float(nv)) if nv.ndim == 0 else nv.ravel()
    if nv.size != size:
        raise InputError("n must be a scalar or have one value per reference SNP")
    if np.any(nv <= 0) or not np.all(np.isfinite(nv)):
        raise InputError("Sample sizes must be finite and > 0")
    sel = _select_rows(rows, size)
    rng = _rng(seed, 0)
    beta_joint = _draw_joint_effects(model, size, rng)
    marginal = _ld_marginal(reference, beta_joint)
    a = _positive_inflation(model.a)
    if a > 0:
        marginal = marginal + rng.standard_normal(size) * np.sqrt(a)
    if replicate is None:
        err = rng.standard_normal(size)
    else:
        err = reference.null_z(int(replicate)).astype(np.float64)
    z = marginal * np.sqrt(nv) + err
    meta = {
        "simulated": {
            "scheme": "ld-reference",
            "reference": reference.name,
            "config": reference.config.id,
            "theta": model.theta.tolist(),
            "seed": seed,
            "replicate": replicate,
            "n_causal": int(np.count_nonzero(beta_joint)),
            "sum_beta_sq": float(beta_joint @ beta_joint),
        }
    }
    return ModelData.from_z(
        z=z[sel],
        n=nv[sel],
        ldscore=reference.ldscore[sel],
        nstar=reference.nstar[sel],
        M=model.M,
        snp=reference.subset_ids(sel),
        ref_index=sel,
        neighborhoods=reference.neighborhoods,
        meta=meta,
    )


# --------------------------------------------------------------------------- QQ plot
@dataclass(frozen=True)
class QQData:
    r"""Observed versus model-expected P-value quantiles (data behind a GENESIS QQ plot).

    Arrays are aligned and sorted by increasing :math:`-\log_{10}P`.

    Attributes
    ----------
    observed
        Sorted observed :math:`-\log_{10}P`.
    expected
        Mean over simulations of the sorted :math:`-\log_{10}P` simulated under the
        fitted model (the x axis of the plot).
    lower, upper
        Point-wise limits of the central ``coverage`` range of the simulated values.
    lambda_obs
        Observed genomic-control factor.
    lambda_mean, lambda_lower, lambda_upper
        Mean and ``coverage`` range of the genomic-control factor across simulations.
    n_sim
        Number of simulations actually run.
    coverage
        Coverage of the envelope.
    n_snps
        Number of markers the quantiles were computed from (before any thinning).
    ld_errors
        ``True`` if the simulations used LD-correlated null errors from the reference;
        ``False`` if independent errors were used (envelope too narrow).
    """

    observed: np.ndarray
    expected: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    lambda_obs: float
    lambda_mean: float
    lambda_lower: float
    lambda_upper: float
    n_sim: int
    coverage: float = 0.8
    n_snps: int = 0
    ld_errors: bool = True

    def thin(self, max_points: int = 5000) -> QQData:
        """Reduce to at most ``max_points`` points for plotting.

        The ``max_points // 2`` most significant points are kept exactly (the tail is
        where the plot is informative); the remainder is thinned evenly.
        """
        idx = _thin_index(self.observed.size, int(max_points))
        return QQData(
            observed=self.observed[idx],
            expected=self.expected[idx],
            lower=self.lower[idx],
            upper=self.upper[idx],
            lambda_obs=self.lambda_obs,
            lambda_mean=self.lambda_mean,
            lambda_lower=self.lambda_lower,
            lambda_upper=self.lambda_upper,
            n_sim=self.n_sim,
            coverage=self.coverage,
            n_snps=self.n_snps,
            ld_errors=self.ld_errors,
        )

    def to_frame(self) -> pd.DataFrame:
        """Return the four aligned curves as a DataFrame (``QQdata`` of the original)."""
        return pd.DataFrame(
            {
                "observed": self.observed,
                "expected": self.expected,
                "lower": self.lower,
                "upper": self.upper,
            }
        )


def qq_simulation(
    data: ModelData,
    fit: Any,
    reference: LDReference | None = None,
    *,
    n_sim: int = 100,
    coverage: float = 0.8,
    seed: int | None = 123,
    c0: int | None = None,
    max_points: int | None = None,
) -> QQData:
    r"""Simulate the P-value distribution expected under the fitted model.

    Port of ``qqplotdata.simu()``. In each of ``n_sim`` replicates every marker gets

    .. math::

        z_k = \sqrt{n_k}\,(\beta_k + \nu_k) + e_k, \qquad
        \beta_k \sim N\bigl(0,\ \tfrac{\ell_k}{N^*_k}\textstyle\sum_h J_h\sigma_h^2\bigr), \quad
        \nu_k \sim N(0, a),

    with the latent counts :math:`J` drawn from the truncated multinomial prior of the
    fitted model and :math:`e` an LD-correlated null Z-score replicate. The sorted
    :math:`-\log_{10}P` values are averaged over replicates (expected curve) and their
    point-wise quantiles give the envelope; the same is done for :math:`\lambda_{GC}`.

    Parameters
    ----------
    data
        The data the model was fitted to.
    fit
        A :class:`~genesis_gwas.result.FitResult` (or a raw parameter vector).
    reference
        The LD reference used to prepare ``data``; supplies the null replicates. Without
        it (or without ``data.ref_index``) independent errors are used and a
        :class:`~genesis_gwas.exceptions.DataWarning` is issued.
    n_sim
        Number of simulations (capped, with a warning, at the number of null replicates
        the reference provides; the bundled European reference has 100).
    coverage
        Coverage of the point-wise envelope (default 0.8, as in the paper's Fig. 1).
    seed
        Seed; replicate ``i`` uses ``default_rng([seed, i])``.
    c0
        Truncation level of the latent counts (default ``fit.c0``, else 10).
    max_points
        Keep only this many order statistics (the most significant half exactly, the
        rest evenly thinned). By default all ``K`` are kept, which needs about
        ``4 * n_sim * K`` bytes of working memory.

    Returns
    -------
    QQData

    Notes
    -----
    Deliberate differences from the original:

    * :math:`\nu_k` is drawn independently for every marker. The original draws a *single*
      ``rnorm(1)`` per replicate and adds that one number to all markers, which shifts
      every Z statistic by a common constant instead of adding the variance :math:`a`.
    * The latent mixture component of each marker is drawn by one vectorised inversion of
      the tabulated truncated prior instead of a per-marker R loop over
      ``sample(1:L, prob = proportions[k, ])``. The distribution is the same.
    * As in the original, a negative ``a`` is treated as zero.
    * P-values are computed in log space, so very large simulated or observed
      :math:`|z|` give finite :math:`-\log_{10}P` rather than infinity.
    """
    if not 0.0 < coverage < 1.0:
        raise InputError("coverage must lie in (0, 1)")
    if int(n_sim) < 1:
        raise InputError("n_sim must be >= 1")
    model = resolve_model(fit)
    if c0 is None:
        c0 = int(fit.c0) if isinstance(fit, FitResult) else 10
    K = data.K
    has_rows = data.ref_index is not None
    if reference is not None and has_rows and int(data.ref_index.max()) >= reference.M:  # type: ignore[union-attr]
        raise InputError("data.ref_index points outside the given reference")
    n_run, ld_errors = _null_source(reference, int(n_sim), "qq_simulation", has_rows)

    z_obs = data.z
    observed = np.sort(_neg_log10_p(z_obs))
    keep = None if max_points is None else _thin_index(K, int(max_points))
    width = K if keep is None else keep.size

    sampler = _ConfigSampler(data.nstar, model, c0)
    scale = data.ldscore / data.nstar
    inv_se = 1.0 / np.sqrt(data.varbetahat)  # sqrt(n_k) for standardised effects
    a = _positive_inflation(model.a)
    sims = np.empty((n_run, width), dtype=np.float32)
    lambdas = np.empty(n_run)
    for i in range(n_run):
        rng = _rng(seed, i)
        effect = rng.standard_normal(K) * np.sqrt(sampler.draw(rng) * scale)
        if a > 0:
            effect += rng.standard_normal(K) * np.sqrt(a)
        if ld_errors:
            assert reference is not None and data.ref_index is not None
            err = reference.null_z(i)[data.ref_index].astype(np.float64)
        else:
            err = rng.standard_normal(K)
        z = effect * inv_se + err
        lambdas[i] = lambda_gc(z)
        nlp = np.sort(_neg_log10_p(z))
        sims[i] = nlp if keep is None else nlp[keep]

    lo_q, hi_q = (1.0 - coverage) / 2.0, 1.0 - (1.0 - coverage) / 2.0
    expected = np.empty(width)
    lower = np.empty(width)
    upper = np.empty(width)
    step = 1 << 17
    for j0 in range(0, width, step):
        block = sims[:, j0 : j0 + step].astype(np.float64)
        expected[j0 : j0 + step] = block.mean(axis=0)
        q = np.quantile(block, [lo_q, hi_q], axis=0)
        lower[j0 : j0 + step] = q[0]
        upper[j0 : j0 + step] = q[1]
    lam_lo, lam_hi = np.quantile(lambdas, [lo_q, hi_q])
    return QQData(
        observed=observed if keep is None else observed[keep],
        expected=expected,
        lower=lower,
        upper=upper,
        lambda_obs=lambda_gc(z_obs),
        lambda_mean=float(lambdas.mean()),
        lambda_lower=float(lam_lo),
        lambda_upper=float(lam_hi),
        n_sim=n_run,
        coverage=float(coverage),
        n_snps=K,
        ld_errors=ld_errors,
    )


# ------------------------------------------------------------------- future lambda GC
def future_gc(
    fit_or_theta: Any,
    n: Any,
    reference: LDReference,
    *,
    n_sim: int = 1,
    seed: int | None = 123,
    M: int | None = None,
) -> Any:
    r"""Predict the genomic-control factor of a future GWAS of sample size ``n``.

    Port of ``futuregc()``: simulate a GWAS on the whole reference panel exactly as
    :func:`simulate_ld_sumstats` does (joint effects from the fitted mixture, marginal
    effects through the tagging neighbourhoods with :math:`|r|`, i.i.d.
    :math:`N(0,a)` confounding, LD-correlated null errors) and return

    .. math::

        \bar\lambda_{GC}(n) = \frac1{n_\text{sim}}\sum_{i}
            \frac{\operatorname{median}_k\, z_{ik}^2}{0.4549}, \qquad
        z_{ik} = \sqrt n\,(\beta_{ik}+\nu_{ik}) + \tilde u_{ik}.

    Parameters
    ----------
    fit_or_theta
        Fitted model or parameter vector. A negative ``a`` is treated as zero, as in the
        original.
    n
        Future sample size; a scalar or an array (the same simulated effects and errors
        are re-used for every sample size, so the curve is smooth in ``n``).
    reference
        LD reference with pairwise :math:`r^2` and null replicates; replicate ``i`` of
        the simulation uses null replicate ``i``.
    n_sim
        Number of simulations to average (the original default is 1).
    seed
        Seed; replicate ``i`` uses ``default_rng([seed, i])``.
    M
        Must equal ``reference.M`` if given: the simulation runs over the reference panel
        itself (the original has the same implicit requirement).

    Returns
    -------
    float or numpy.ndarray
        Mean predicted :math:`\lambda_{GC}`, with the shape of ``n``.
    """
    if M is not None and int(M) != reference.M:
        raise InputError(
            f"M = {M} differs from the reference panel size {reference.M}; the simulation runs "
            "over the reference SNPs themselves."
        )
    if int(n_sim) < 1:
        raise InputError("n_sim must be >= 1")
    model = resolve_model(fit_or_theta, reference.M)
    scalar = np.ndim(n) == 0
    nv = np.atleast_1d(np.asarray(n, dtype=np.float64)).ravel()
    if nv.size == 0 or np.any(nv <= 0) or not np.all(np.isfinite(nv)):
        raise InputError("n must contain finite sample sizes > 0")
    n_run, ld_errors = _null_source(reference, int(n_sim), "future_gc", True)
    size = reference.M
    a = _positive_inflation(model.a)
    lam = np.empty((n_run, nv.size))
    for i in range(n_run):
        rng = _rng(seed, i)
        marginal = _ld_marginal(reference, _draw_joint_effects(model, size, rng))
        if a > 0:
            marginal += rng.standard_normal(size) * np.sqrt(a)
        err = reference.null_z(i).astype(np.float64) if ld_errors else rng.standard_normal(size)
        for j, nj in enumerate(nv):
            lam[i, j] = lambda_gc(marginal * np.sqrt(nj) + err)
    out = lam.mean(axis=0)
    return float(out[0]) if scalar else out.reshape(np.shape(n))
