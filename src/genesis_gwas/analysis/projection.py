r"""Projecting the yield of future GWAS from a fitted effect-size distribution.

For a future study of (effective) sample size :math:`n`, a susceptibility SNP with
standardised joint effect :math:`\beta` reaches significance level :math:`\alpha` with
probability

.. math::

    \mathrm{pow}_\alpha(\beta) = 1-\Phi\bigl(c_{\alpha}-\sqrt{n}\,\beta\bigr)
                               + \Phi\bigl(-c_{\alpha}-\sqrt{n}\,\beta\bigr),
    \qquad c_\alpha = \Phi^{-1}(1-\alpha/2).

Zhang et al. (2018, Methods, "Future projection") project

.. math::

    E(\mathrm{ND}_\alpha) &= M\pi_c \int \mathrm{pow}_\alpha(\beta)\, f(\beta)\, d\beta
        && \text{(number of discoveries)} \\
    E(\mathrm{GV}_\alpha) &= \frac{M\pi_c}{h^2} \int \beta^2\,\mathrm{pow}_\alpha(\beta)\, f(\beta)\, d\beta
        && \text{(fraction of } h^2 \text{ explained by them)}

with :math:`f` the normal-mixture density of susceptibility-SNP effects and
:math:`h^2 = M\pi_c\sum_h p_h\sigma_h^2`. The original ``projection()`` evaluates both
integrals by adaptive quadrature. They have closed forms, which this module uses.

Closed forms
------------
Write :math:`\mathrm{pow}_\alpha(\beta)=\Pr(|W|>c_\alpha\mid\beta)` with
:math:`W=\sqrt n\,\beta+Z`, :math:`Z\sim N(0,1)`. If :math:`\beta\sim N(0,\sigma^2)` then
:math:`W\sim N(0,\tau^2)`, :math:`\tau^2=1+n\sigma^2`, and :math:`(\beta, W)` is bivariate
normal with :math:`E[\beta^2\mid W]=\sigma^2/\tau^2+(n\sigma^4/\tau^4)W^2`. With
:math:`t=c_\alpha/\tau`:

.. math::

    \int \mathrm{pow}_\alpha(\beta)\,N(\beta;0,\sigma^2)\,d\beta &= 2\,\Phi(-t), \\
    \int \beta^2\,\mathrm{pow}_\alpha(\beta)\,N(\beta;0,\sigma^2)\,d\beta
        &= 2\sigma^2\Bigl[\Phi(-t)+\frac{n\sigma^2}{1+n\sigma^2}\,t\,\varphi(t)\Bigr].

Both are exact, vectorise over :math:`n`, and agree with the quadrature of the original
to its integration tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit, logit, ndtr, ndtri

from genesis_gwas.analysis.density import EffectModel, resolve_model
from genesis_gwas.exceptions import InputError
from genesis_gwas.result import FitResult

__all__ = ["Projection", "power_integrals", "project"]

_SQRT_2PI = float(np.sqrt(2.0 * np.pi))


@dataclass(frozen=True)
class Projection:
    r"""Projected yield of future GWAS at one or more sample sizes.

    All array attributes have one entry per requested sample size.

    Attributes
    ----------
    n
        Sample sizes.
    alpha
        Significance level for declaring a discovery.
    M
        Number of reference SNPs.
    heritability
        Total heritability :math:`h^2` under the fitted model.
    n_discoveries
        Expected number of susceptibility SNPs reaching significance,
        :math:`E(\mathrm{ND}_\alpha)`.
    gv_percentage
        Percentage of :math:`h^2` explained by those SNPs, :math:`100\,E(\mathrm{GV}_\alpha)`.
    pheno_variance
        The same on the absolute scale, ``gv_percentage * heritability / 100`` (phenotypic
        variance explained for a quantitative trait).
    n_discoveries_lower, n_discoveries_upper, gv_percentage_lower, gv_percentage_upper, pheno_variance_lower, pheno_variance_upper
        Limits of the simulation-based confidence intervals (``None`` unless
        ``ci=True``).
    coverage, n_sim
        Coverage and number of parameter draws of the intervals (``None`` unless
        ``ci=True``).
    """

    n: np.ndarray
    alpha: float
    M: int
    heritability: float
    n_discoveries: np.ndarray
    gv_percentage: np.ndarray
    pheno_variance: np.ndarray
    n_discoveries_lower: np.ndarray | None = None
    n_discoveries_upper: np.ndarray | None = None
    gv_percentage_lower: np.ndarray | None = None
    gv_percentage_upper: np.ndarray | None = None
    pheno_variance_lower: np.ndarray | None = None
    pheno_variance_upper: np.ndarray | None = None
    coverage: float | None = None
    n_sim: int | None = None

    @property
    def has_ci(self) -> bool:
        """Whether confidence limits are available."""
        return self.n_discoveries_lower is not None

    def to_frame(self) -> pd.DataFrame:
        """Tidy table with one row per sample size (interval columns only if computed)."""
        cols: dict[str, Any] = {
            "n": self.n,
            "n_discoveries": self.n_discoveries,
            "gv_percentage": self.gv_percentage,
            "pheno_variance": self.pheno_variance,
        }
        if self.has_ci:
            cols = {
                "n": self.n,
                "n_discoveries": self.n_discoveries,
                "n_discoveries_lower": self.n_discoveries_lower,
                "n_discoveries_upper": self.n_discoveries_upper,
                "gv_percentage": self.gv_percentage,
                "gv_percentage_lower": self.gv_percentage_lower,
                "gv_percentage_upper": self.gv_percentage_upper,
                "pheno_variance": self.pheno_variance,
                "pheno_variance_lower": self.pheno_variance_lower,
                "pheno_variance_upper": self.pheno_variance_upper,
            }
        frame = pd.DataFrame(cols)
        frame.insert(1, "heritability", self.heritability)
        return frame


def power_integrals(n: Any, variance: Any, alpha: float = 5e-8) -> tuple[np.ndarray, np.ndarray]:
    r"""Closed-form power integrals for a normal effect-size component.

    For :math:`\beta\sim N(0,\sigma^2)`, sample size :math:`n` and two-sided level
    :math:`\alpha`, with :math:`t=c_\alpha/\sqrt{1+n\sigma^2}`:

    .. math::

        E[\mathrm{pow}_\alpha(\beta)] = 2\Phi(-t), \qquad
        E[\beta^2\,\mathrm{pow}_\alpha(\beta)]
            = 2\sigma^2\Bigl[\Phi(-t)+\frac{n\sigma^2}{1+n\sigma^2}\,t\,\varphi(t)\Bigr].

    Parameters
    ----------
    n
        Sample size(s).
    variance
        Effect-size variance(s) :math:`\sigma^2`; broadcast against ``n``.
    alpha
        Two-sided significance level.

    Returns
    -------
    (expected_power, expected_beta2_power)
        Arrays with the broadcast shape of ``n`` and ``variance``.
    """
    if not 0.0 < alpha < 1.0:
        raise InputError("alpha must lie in (0, 1)")
    nv = np.asarray(n, dtype=np.float64)
    s2 = np.asarray(variance, dtype=np.float64)
    if np.any(nv <= 0) or not np.all(np.isfinite(nv)):
        raise InputError("Sample sizes must be finite and > 0")
    if np.any(s2 < 0) or not np.all(np.isfinite(s2)):
        raise InputError("Variances must be finite and >= 0")
    c = -float(ndtri(alpha / 2.0))
    with np.errstate(over="ignore", invalid="ignore"):
        tau2 = 1.0 + nv * s2
        # n sigma^2 / (1 + n sigma^2); its limit 1 when the product overflows
        ratio = np.where(np.isfinite(tau2), nv * s2 / tau2, 1.0)
    t = c / np.sqrt(tau2)
    tail = ndtr(-t)
    phi = np.exp(-0.5 * t * t) / _SQRT_2PI
    return 2.0 * tail, 2.0 * s2 * (tail + ratio * t * phi)


def _point(
    pic_values: Any, weights: Any, variances: Any, n: np.ndarray, alpha: float, M: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Discoveries, variance explained and heritability for (batches of) parameter values.

    ``pic_values`` has shape ``B``; ``weights`` and ``variances`` have shape ``B + (H,)``;
    the outputs have shape ``B + (len(n),)`` (heritability: ``B``).
    """
    pic = np.asarray(pic_values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    s2 = np.asarray(variances, dtype=np.float64)
    e_pow, e_b2 = power_integrals(n[:, None], s2[..., None, :], alpha)  # B + (n, H)
    nd = M * pic[..., None] * np.sum(w[..., None, :] * e_pow, axis=-1)
    explained = M * pic[..., None] * np.sum(w[..., None, :] * e_b2, axis=-1)
    herit = M * pic * np.sum(w * s2, axis=-1)
    return nd, explained, herit


def project(
    fit_or_theta: Any,
    n: Any,
    *,
    alpha: float = 5e-8,
    M: int | None = None,
    covariance: Any = None,
    ci: bool = False,
    n_sim: int = 1000,
    coverage: float = 0.95,
    seed: int | None = 123,
) -> Projection:
    r"""Project the number of discoveries and heritability explained in a future GWAS.

    Port of ``projection()`` of the original package (point estimates identical to its
    quadrature; see the module docstring for the closed forms).

    Parameters
    ----------
    fit_or_theta
        Fitted model, or parameter vector ``(pic, sigmasq, a)`` /
        ``(pic, p1, sigmasq1, sigmasq2, a)``.
    n
        Future (effective) sample size(s); scalar or array.
    alpha
        Genome-wide significance level (default :math:`5\times10^{-8}`).
    M
        Number of reference SNPs (default: that of the fit, else 1,070,777).
    covariance
        Covariance matrix of the parameter estimates in the layout of the parameter
        vector (default: ``fit.covariance``). Needed only for ``ci=True``.
    ci
        Also compute confidence intervals by simulation (below).
    n_sim
        Number of parameter draws for the intervals.
    coverage
        Coverage of the intervals (default 0.95).
    seed
        Seed for :func:`numpy.random.default_rng`.

    Returns
    -------
    Projection

    Notes
    -----
    **Confidence intervals.** As in the original, uncertainty is propagated by drawing
    parameter values from a normal approximation on a transformed scale, recomputing the
    projection for each draw, adding binomial sampling variation to the discovery count
    (``Binomial(M, ND/M)``), and taking empirical quantiles; the interval for
    ``pheno_variance`` is the interval for ``gv_percentage`` scaled by the point estimate
    of :math:`h^2`. Two details deliberately differ from the original:

    * The original draws *all* parameters, including ``a``, on the log scale. ``log(a)``
      is undefined for the (legitimate) negative estimates of ``a`` and makes the original
      fail; the projection does not depend on ``a``, so it is left out here.
    * The original also draws ``p1`` on the log scale, which can produce ``p1 > 1`` and
      hence negative mixture weights. Here ``pic`` and the variances are drawn on the log
      scale and ``p1`` on the logit scale, with the delta-method covariance
      :math:`D V D^\top`, :math:`D=\operatorname{diag}\{1/\pi_c,\;1/[p_1(1-p_1)],\;1/\sigma_h^2\}`.

    Because the random-number generators differ, interval limits agree with the original
    only up to Monte-Carlo error.
    """
    if isinstance(fit_or_theta, FitResult) and covariance is None:
        covariance = fit_or_theta.covariance
    model = resolve_model(fit_or_theta, M)
    nv = np.atleast_1d(np.asarray(n, dtype=np.float64)).ravel()
    if nv.size == 0:
        raise InputError("n must contain at least one sample size")
    nd, explained, herit = _point(model.pic, model.weights, model.variances, nv, alpha, model.M)
    gv = 100.0 * explained / herit
    if not ci:
        return Projection(
            n=nv,
            alpha=float(alpha),
            M=model.M,
            heritability=float(herit),
            n_discoveries=nd,
            gv_percentage=gv,
            pheno_variance=explained,
        )

    if covariance is None:
        raise InputError("ci=True needs the covariance matrix of the parameter estimates")
    if not 0.0 < coverage < 1.0:
        raise InputError("coverage must lie in (0, 1)")
    if int(n_sim) < 2:
        raise InputError("n_sim must be >= 2")
    draws_pic, draws_w, draws_s2 = _draw_parameters(model, covariance, int(n_sim), seed)
    nd_s, explained_s, herit_s = _point(draws_pic, draws_w, draws_s2, nv, alpha, model.M)
    gv_s = 100.0 * explained_s / herit_s[:, None]
    rng = np.random.default_rng(None if seed is None else [int(seed), 1])
    counts = rng.binomial(model.M, np.clip(nd_s / model.M, 0.0, 1.0))
    lo_q, hi_q = (1.0 - coverage) / 2.0, 1.0 - (1.0 - coverage) / 2.0
    nd_lo, nd_hi = np.quantile(counts, [lo_q, hi_q], axis=0)
    gv_lo, gv_hi = np.quantile(gv_s, [lo_q, hi_q], axis=0)
    return Projection(
        n=nv,
        alpha=float(alpha),
        M=model.M,
        heritability=float(herit),
        n_discoveries=nd,
        gv_percentage=gv,
        pheno_variance=explained,
        n_discoveries_lower=nd_lo,
        n_discoveries_upper=nd_hi,
        gv_percentage_lower=gv_lo,
        gv_percentage_upper=gv_hi,
        pheno_variance_lower=gv_lo * float(herit) / 100.0,
        pheno_variance_upper=gv_hi * float(herit) / 100.0,
        coverage=float(coverage),
        n_sim=int(n_sim),
    )


def _draw_parameters(
    model: EffectModel, covariance: Any, n_sim: int, seed: int | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw ``(pic, weights, variances)`` from the delta-method normal on the working scale."""
    cov = np.asarray(covariance, dtype=np.float64)
    P = model.theta.size
    if cov.shape != (P, P):
        raise InputError(f"covariance must have shape ({P}, {P}) to match the parameter vector")
    keep = np.arange(P - 1)  # everything except `a`
    sub = cov[np.ix_(keep, keep)]
    if not np.all(np.isfinite(sub)):
        raise InputError("covariance contains non-finite values; confidence intervals unavailable")
    sub = 0.5 * (sub + sub.T)
    theta = model.theta[keep]
    if model.n_components == 2:
        mean = np.log(theta)
        jac = 1.0 / theta
    else:
        p1 = float(theta[1])
        if not 0.0 < p1 < 1.0:
            raise InputError("p1 must lie strictly inside (0, 1) to compute confidence intervals")
        mean = np.array([np.log(theta[0]), logit(p1), np.log(theta[2]), np.log(theta[3])])
        jac = np.array([1.0 / theta[0], 1.0 / (p1 * (1.0 - p1)), 1.0 / theta[2], 1.0 / theta[3]])
    work = sub * jac[:, None] * jac[None, :]
    evals, evecs = np.linalg.eigh(work)
    root = evecs * np.sqrt(np.clip(evals, 0.0, None))[None, :]
    rng = np.random.default_rng(None if seed is None else [int(seed), 0])
    g = mean[None, :] + rng.standard_normal((n_sim, mean.size)) @ root.T
    # exp() of an extreme draw may overflow to inf for a hopelessly uncertain fit; cap the
    # working scale where exp is still finite (a variance of e^50 is equally "everything").
    g_log = np.clip(g, -700.0, 50.0)
    pic = np.minimum(np.exp(g_log[:, 0]), 1.0)
    if model.n_components == 2:
        weights = np.ones((n_sim, 1))
        variances = np.exp(g_log[:, 1:2])
    else:
        p1_draw = expit(g[:, 1])
        weights = np.stack([p1_draw, 1.0 - p1_draw], axis=1)
        variances = np.exp(g_log[:, 2:4])
    return pic, weights, variances
