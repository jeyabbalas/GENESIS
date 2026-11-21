r"""Expected performance of polygenic risk scores built from a future GWAS.

This is a port of ``polyriskpredict()`` of the original package, which implements the
theory of Chatterjee et al. (2013, *Nat Genet* 45:400) for a polygenic score

.. math::

    \hat S = \sum_{k:\,|z_k| > c} \hat\beta_k\, G_k

built from the SNPs of a GWAS of size :math:`N` whose association statistics exceed the
threshold :math:`c = \Phi^{-1}(1-\alpha/2)`, under the assumption of ``M`` independent
SNPs of which ``M1`` are susceptibility SNPs with effects
:math:`\beta\sim\sum_h p_h N(0,\sigma_h^2)`.

The predictive performance is summarised by

.. math::

    \mu(c) \;=\; \frac{\operatorname{cov}(\hat S, S)}{\operatorname{sd}(\hat S)}
          \;=\; \frac{C_1 + C_4}{\sqrt{C_1 + C_2 + C_3 + C_4 + C_5}},

the mean difference of the standardised score between cases and controls on the
log-relative-risk scale. With :math:`\tau_h^2 = 1+N\sigma_h^2` and :math:`t_h = c/\tau_h`
the five terms are

.. math::

    C_1 &= M_1\textstyle\sum_h p_h\; 2\sigma_h^2\bigl[\Phi(-t_h) + \tfrac{N\sigma_h^2}{\tau_h^2}\,t_h\varphi(t_h)\bigr]
        && E[\beta^2\,\mathbb 1\{\text{selected}\}] \\
    C_2 &= M_1\textstyle\sum_h p_h\; 2\Phi(-t_h)/N \\
    C_3 &= M_1\textstyle\sum_h p_h\; 2\,t_h\varphi(t_h)/N \\
    C_4 &= M_1\textstyle\sum_h p_h\; 2\sigma_h^2\,t_h\varphi(t_h)/\tau_h^2
        && E[\beta\,e\,\mathbb 1\{\text{selected}\}]/\sqrt N \\
    C_5 &= (M-M_1)\,\bigl[2\Phi(-c) + 2c\,\varphi(c)\bigr]/N
        && \text{noise from null SNPs.}

:math:`C_1+C_4` is the covariance between the score and the true genetic value and the
sum of all five is the variance of the score. These are algebraically the
``comp1``–``comp4``/``cmp5`` expressions of the original code, written with upper-tail
probabilities so that nothing cancels at stringent thresholds.

From :math:`\mu` at the chosen threshold,

.. math::

    \mathrm{AUC} = \Phi\bigl(\mu/\sqrt2\bigr),

and for a log-normal risk model with standard deviation :math:`s` of the log relative
risk, the proportions of the *population* (PPI) and of future *cases* (PCI) whose risk is
at least :math:`k` times the population average are

.. math::

    \mathrm{PPI}_k = 1-\Phi\bigl(\tfrac{\log k}{s}+\tfrac{s}{2}\bigr), \qquad
    \mathrm{PCI}_k = 1-\Phi\bigl(\tfrac{\log k}{s}-\tfrac{s}{2}\bigr).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import ndtr, ndtri

from genesis_gwas.analysis.density import DEFAULT_M, resolve_model
from genesis_gwas.exceptions import GenesisError, InputError

__all__ = ["PolygenicPrediction", "polyrisk_mu", "polyrisk_predict"]

_SQRT_2PI = math.sqrt(2.0 * math.pi)
_GRID_SIZE = 1000
_GRID_LOW = 1e-8
_GRID_HIGH = 7.0
_OPT_TOL = 1e-5


@dataclass(frozen=True)
class PolygenicPrediction:
    r"""Expected performance of a polygenic risk score.

    For a scalar sample size the attributes ``alpha``, ``auc`` and ``mu`` are floats and
    ``ppi``/``pci`` have shape ``(len(k_fold),)``; for an array of sample sizes they gain
    a leading dimension.

    Attributes
    ----------
    n
        GWAS sample size(s) at which the score is built.
    threshold
        ``"optimum"`` (the threshold maximising :math:`\mu`) or ``"gwas"``.
    alpha
        Two-sided significance level for SNP inclusion: the optimal level, or
        ``alpha_gwas``.
    mu
        :math:`\mu = \operatorname{cov}(\hat S,S)/\operatorname{sd}(\hat S)` at that level.
    auc
        Area under the ROC curve, :math:`\Phi(\mu/\sqrt2)`.
    k_fold
        Risk multiples at which ``ppi`` and ``pci`` are evaluated.
    ppi
        Proportion of the population at :math:`k`-fold or higher risk than average.
    pci
        Proportion of cases arising from that part of the population.
    legacy
        Whether ``ppi``/``pci`` use the original code's :math:`s=\sqrt\mu` (see
        :func:`polyrisk_predict`).
    """

    n: Any
    threshold: str
    alpha: Any
    mu: Any
    auc: Any
    k_fold: np.ndarray
    ppi: np.ndarray
    pci: np.ndarray
    legacy: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Plain-Python representation (arrays become lists)."""

        def conv(x: Any) -> Any:
            arr = np.asarray(x)
            return arr.tolist()

        return {
            "n": conv(self.n),
            "threshold": self.threshold,
            "alpha": conv(self.alpha),
            "mu": conv(self.mu),
            "auc": conv(self.auc),
            "k_fold": conv(self.k_fold),
            "ppi": conv(self.ppi),
            "pci": conv(self.pci),
            "legacy": self.legacy,
        }


def polyrisk_mu(c: Any, n: float, ps: Any, sig2s: Any, M: float, M1: float) -> np.ndarray:
    r"""Compute the performance measure :math:`\mu(c)` of a thresholded polygenic score.

    Port of ``mu.func`` of the original ``polyriskpredict()`` (see the module docstring
    for the formula); vectorised over the threshold ``c``.

    Parameters
    ----------
    c
        Threshold(s) on the absolute Z statistic, :math:`c=\Phi^{-1}(1-\alpha/2)`.
    n
        GWAS sample size.
    ps, sig2s
        Mixing weights and variances of the effect-size distribution of susceptibility
        SNPs.
    M, M1
        Number of independent SNPs and number of susceptibility SNPs among them.
    """
    cv = np.asarray(c, dtype=np.float64)
    p = np.asarray(ps, dtype=np.float64)
    s2 = np.asarray(sig2s, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        tau2 = 1.0 + n * s2
        # n sigma^2 / (1 + n sigma^2); its limit 1 when the product overflows
        ratio = np.where(np.isfinite(tau2), n * s2 / tau2, 1.0)
    tau = np.sqrt(tau2)
    t = cv[..., None] / tau
    tail = ndtr(-t)
    tphi = t * np.exp(-0.5 * t * t) / _SQRT_2PI
    cmp1 = M1 * ((2.0 * s2 * (tail + ratio * tphi)) @ p)
    cmp2 = M1 * ((2.0 * tail / n) @ p)
    cmp3 = M1 * ((2.0 * tphi / n) @ p)
    cmp4 = M1 * ((2.0 * s2 * tphi / tau2) @ p)
    cmp5 = (M - M1) * (2.0 * ndtr(-cv) + 2.0 * cv * np.exp(-0.5 * cv * cv) / _SQRT_2PI) / n
    return np.asarray((cmp1 + cmp4) / np.sqrt(cmp1 + cmp2 + cmp3 + cmp4 + cmp5))


def _brent_fmin(f: Callable[[float], float], ax: float, bx: float, tol: float) -> float:
    """Minimise ``f`` on ``[ax, bx]`` — a literal port of R's ``Brent_fmin`` (``optimize``).

    Porting the routine, rather than calling a different bounded minimiser, makes the
    optimal threshold agree with the original far below the optimiser tolerance.
    """
    golden = (3.0 - math.sqrt(5.0)) * 0.5
    eps = math.sqrt(float(np.finfo(np.float64).eps))
    a, b = ax, bx
    v = a + golden * (b - a)
    w = v
    x = v
    d = 0.0
    e = 0.0
    fx = f(x)
    fv = fx
    fw = fx
    tol3 = tol / 3.0
    for _ in range(1000):
        xm = (a + b) * 0.5
        tol1 = eps * abs(x) + tol3
        t2 = tol1 * 2.0
        if abs(x - xm) <= t2 - (b - a) * 0.5:
            return x
        p = q = r = 0.0
        if abs(e) > tol1:  # fit a parabola
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = (q - r) * 2.0
            if q > 0.0:
                p = -p
            else:
                q = -q
            r = e
            e = d
        if abs(p) >= abs(q * 0.5 * r) or p <= q * (a - x) or p >= q * (b - x):
            e = b - x if x < xm else a - x  # golden-section step
            d = golden * e
        else:  # parabolic-interpolation step
            d = p / q
            u = x + d
            if u - a < t2 or b - u < t2:
                d = tol1
                if x >= xm:
                    d = -d
        if abs(d) >= tol1:
            u = x + d
        elif d > 0.0:
            u = x + tol1
        else:
            u = x - tol1
        fu = f(u)
        if not math.isfinite(fu):
            raise GenesisError("Non-finite objective while optimising the PRS threshold")
        if fu <= fx:
            if u < x:
                b = x
            else:
                a = x
            v, w, x = w, x, u
            fv, fw, fx = fw, fx, fu
        else:
            if u < x:
                a = u
            else:
                b = u
            if fu <= fw or w == x:
                v, fv = w, fw
                w, fw = u, fu
            elif fu <= fv or v in (x, w):
                v, fv = u, fu
    raise GenesisError("Threshold optimisation did not converge")  # pragma: no cover


def _optimal_threshold(
    n: float, ps: np.ndarray, sig2s: np.ndarray, M: float, M1: float
) -> tuple[float, float]:
    """Threshold ``c`` maximising ``mu`` and the maximum, found exactly as the original does."""
    grid = np.exp(np.linspace(math.log(_GRID_LOW), math.log(_GRID_HIGH), _GRID_SIZE))
    mus = polyrisk_mu(grid, n, ps, sig2s, M, M1)
    if not np.all(np.isfinite(mus)):
        raise GenesisError("The PRS performance measure is not finite; check the inputs")
    i = int(np.argmax(mus))
    if i == 0:
        return float(grid[0]), float(mus[0])
    upper = float(grid[min(i + 1, _GRID_SIZE - 1)])
    c_opt = _brent_fmin(
        lambda x: -float(polyrisk_mu(x, n, ps, sig2s, M, M1)), float(grid[i - 1]), upper, _OPT_TOL
    )
    return c_opt, float(polyrisk_mu(c_opt, n, ps, sig2s, M, M1))


def polyrisk_predict(
    n: Any,
    fit_or_theta: Any = None,
    *,
    ps: Any = None,
    sig2s: Any = None,
    M: float | None = None,
    M1: float | None = None,
    threshold: str = "optimum",
    alpha_gwas: float = 5e-8,
    k_fold: Sequence[float] = (3, 4, 5),
    legacy: bool = False,
) -> PolygenicPrediction:
    r"""Predict the AUC and risk stratification achievable with a polygenic risk score.

    Port of ``polyriskpredict()`` (Fig. 4 and Supplementary Tables 10–11 of Zhang et
    al. 2018). SNPs enter the score either at the significance threshold that maximises
    predictive performance or at a fixed genome-wide level.

    Parameters
    ----------
    n
        Sample size(s) of the GWAS used to build the score. For a binary trait this is
        the effective sample size on which the effect sizes are standardised.
    fit_or_theta
        Fitted model or parameter vector. It supplies the effect-size distribution
        (``ps``, ``sig2s``) and the number of susceptibility SNPs ``M1 = M * pic``;
        explicit ``ps``/``sig2s``/``M1`` override it. For the two-component model the
        single normal is used, i.e. ``sig2s = sigmasq``. (The worked example shipped
        with the original package passes ``est[3]``, the inflation parameter ``a``,
        where the variance ``est[2]`` is meant; that slip is not reproduced.)
    ps, sig2s
        Mixing weights (summing to one) and variances of the effect-size distribution of
        susceptibility SNPs. The original requires exactly two components; any number
        :math:`\ge 1` is accepted here.
    M
        Number of independent SNPs (default: ``M`` of the fit, else 1,070,777). The
        paper's Fig. 4 uses 200,000 "independent sets of SNPs".
    M1
        Number of susceptibility SNPs. Defaults to the fitted number, ``fit.M * pic``
        (it does not change when ``M`` is overridden); required without a fitted model.
    threshold
        ``"optimum"``: maximise :math:`\mu(c)` over the threshold — a 1000-point
        logarithmic grid on :math:`c\in[10^{-8},7]` followed by Brent's method between
        the neighbours of the best grid point (tolerance :math:`10^{-5}`), exactly as the
        original. ``"gwas"``: use ``alpha_gwas``.
    alpha_gwas
        Genome-wide significance level for ``threshold="gwas"``.
    k_fold
        Risk multiples for ``ppi`` and ``pci``.
    legacy
        Selects the standard deviation :math:`s` of the log relative risk used in the
        PPI/PCI formulas.

        ``False`` (default)
            :math:`s=\mu`. Under the log-normal risk model that gives
            :math:`\mathrm{AUC}=\Phi(\mu/\sqrt2)`, the calibrated score
            :math:`E[S\mid\hat S]` has standard deviation
            :math:`\operatorname{cov}(\hat S,S)/\operatorname{sd}(\hat S)=\mu`, so this is
            the choice consistent with the reported AUC.
        ``True``
            :math:`s=\sqrt{\mu}`, exactly as coded in the original ``polyriskpredict()``
            (and hence as in the paper's Supplementary Tables 10–11). Because
            :math:`\mu<1` in practice, :math:`\sqrt\mu>\mu` and the original overstates
            the spread of risk, i.e. it yields larger PPI/PCI.

    Returns
    -------
    PolygenicPrediction

    Raises
    ------
    InputError
        For inconsistent or out-of-range inputs.

    Notes
    -----
    The original stops with an R error when the best grid point is the last one
    (:math:`c=7`); here the search interval is then closed at the grid end.
    """
    if threshold not in ("optimum", "gwas"):
        raise InputError("threshold must be 'optimum' or 'gwas'")
    if not 0.0 < alpha_gwas < 1.0:
        raise InputError("alpha_gwas must lie in (0, 1)")
    kf = np.atleast_1d(np.asarray(k_fold, dtype=np.float64)).ravel()
    if kf.size == 0 or np.any(kf <= 0) or not np.all(np.isfinite(kf)):
        raise InputError("k_fold must contain positive numbers")

    if fit_or_theta is not None:
        model = resolve_model(fit_or_theta)
        m_total = float(model.M) if M is None else float(M)
        p_arr = model.weights if ps is None else np.asarray(ps, dtype=np.float64).ravel()
        s_arr = model.variances if sig2s is None else np.asarray(sig2s, dtype=np.float64).ravel()
        m1 = model.n_ssnp if M1 is None else float(M1)
    else:
        if ps is None or sig2s is None or M1 is None:
            raise InputError("Without a fitted model, ps, sig2s and M1 are all required")
        m_total = float(DEFAULT_M) if M is None else float(M)
        p_arr = np.asarray(ps, dtype=np.float64).ravel()
        s_arr = np.asarray(sig2s, dtype=np.float64).ravel()
        m1 = float(M1)
    if p_arr.size != s_arr.size or p_arr.size == 0:
        raise InputError("ps and sig2s must have the same, non-zero, length")
    if np.any(p_arr < 0) or not np.isclose(p_arr.sum(), 1.0, rtol=0, atol=1e-8):
        raise InputError("ps must be non-negative and sum to one")
    if np.any(s_arr < 0) or not np.all(np.isfinite(s_arr)):
        raise InputError("sig2s must be finite and >= 0")
    if not (m_total > 0 and 0 < m1 <= m_total):
        raise InputError("Need 0 < M1 <= M")

    scalar = np.ndim(n) == 0
    nv = np.atleast_1d(np.asarray(n, dtype=np.float64)).ravel()
    if nv.size == 0 or np.any(nv <= 0) or not np.all(np.isfinite(nv)):
        raise InputError("n must contain finite sample sizes > 0")

    alphas = np.empty(nv.size)
    mus = np.empty(nv.size)
    for i, ni in enumerate(nv):
        if threshold == "optimum":
            c_opt, mu = _optimal_threshold(float(ni), p_arr, s_arr, m_total, m1)
            alphas[i] = 2.0 * float(ndtr(-c_opt))
        else:
            c_gwas = -float(ndtri(alpha_gwas / 2.0))
            mu = float(polyrisk_mu(c_gwas, float(ni), p_arr, s_arr, m_total, m1))
            alphas[i] = alpha_gwas
        if not math.isfinite(mu):
            raise GenesisError(f"The PRS performance measure is not finite for n = {ni:g}")
        mus[i] = mu
    auc = ndtr(mus / math.sqrt(2.0))
    s = np.sqrt(mus) if legacy else mus
    with np.errstate(divide="ignore", invalid="ignore"):
        xi = np.log(kf)[None, :] / s[:, None]
    # s = 0 (a score without any predictive value): everybody is at exactly average risk
    xi = np.where(np.isnan(xi), -np.inf, xi)
    ppi = ndtr(-(xi + s[:, None] / 2.0))
    pci = ndtr(-(xi - s[:, None] / 2.0))
    if scalar:
        return PolygenicPrediction(
            n=float(nv[0]),
            threshold=threshold,
            alpha=float(alphas[0]),
            mu=float(mus[0]),
            auc=float(auc[0]),
            k_fold=kf,
            ppi=ppi[0],
            pci=pci[0],
            legacy=bool(legacy),
        )
    return PolygenicPrediction(
        n=nv,
        threshold=threshold,
        alpha=alphas,
        mu=mus,
        auc=auc,
        k_fold=kf,
        ppi=ppi,
        pci=pci,
        legacy=bool(legacy),
    )
