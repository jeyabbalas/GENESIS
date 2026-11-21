r"""The fitted effect-size distribution: density, distribution function, tail counts.

GENESIS models the *joint* (LD-adjusted) standardised effect of each of the :math:`M`
reference SNPs as a spike-and-slab mixture (Zhang et al. 2018, eq. 1)

.. math::

    \beta \;\sim\; \pi_c \sum_{h=1}^{H} p_h\, N(0, \sigma_h^2) \;+\; (1-\pi_c)\,\delta_0 .

The functions in this module describe the *slab*, i.e. the distribution of the effect of
a **susceptibility SNP**,

.. math::

    f(\beta) = \sum_{h=1}^{H} p_h\, \frac{1}{\sigma_h}\,\varphi\!\Bigl(\frac{\beta}{\sigma_h}\Bigr),
    \qquad
    F(\beta) = \sum_{h=1}^{H} p_h\, \Phi\!\Bigl(\frac{\beta}{\sigma_h}\Bigr),

with :math:`H=1` for the two-component model (M2) and :math:`H=2` for the
three-component model (M3). They are the counterparts of ``dmixssnp()`` and
``numInterval()`` in the original R package.

Every function accepts either a :class:`~genesis_gwas.result.FitResult` or a raw
parameter vector: three values ``(pic, sigmasq, a)`` are read as M2 and five values
``(pic, p1, sigmasq1, sigmasq2, a)`` as M3. :func:`resolve_model` implements that
convention for the whole :mod:`genesis_gwas.analysis` package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import ndtr

from genesis_gwas.core.spec import ModelSpec
from genesis_gwas.exceptions import InputError
from genesis_gwas.result import FitResult

__all__ = [
    "DEFAULT_M",
    "EffectModel",
    "effect_cdf",
    "effect_density",
    "effect_mixture",
    "n_in_interval",
    "resolve_model",
]

#: Number of SNPs in the bundled HapMap3 / 1000 Genomes EUR reference panel; the default
#: ``M`` of the original software.
DEFAULT_M = 1_070_777

_SQRT_2PI = float(np.sqrt(2.0 * np.pi))


@dataclass(frozen=True)
class EffectModel:
    r"""A fitted effect-size model in a form convenient for downstream calculations.

    Attributes
    ----------
    n_components
        ``2`` (M2) or ``3`` (M3).
    theta
        The parameter vector, ``(pic, sigmasq, a)`` or ``(pic, p1, sigmasq1, sigmasq2, a)``.
    pic
        Proportion of susceptibility SNPs :math:`\pi_c`.
    weights
        ``(H,)`` mixing weights :math:`p_h` among susceptibility SNPs (sum to one).
    variances
        ``(H,)`` variance components :math:`\sigma_h^2`.
    a
        Variance-inflation parameter (not part of the effect-size distribution).
    M
        Number of SNPs in the reference panel.
    """

    n_components: int
    theta: np.ndarray
    pic: float
    weights: np.ndarray
    variances: np.ndarray
    a: float
    M: int

    @property
    def spec(self) -> ModelSpec:
        """The corresponding :class:`~genesis_gwas.core.spec.ModelSpec`."""
        return ModelSpec(self.n_components)

    @property
    def sds(self) -> np.ndarray:
        """``(H,)`` standard deviations :math:`\\sigma_h`."""
        return np.sqrt(self.variances)

    @property
    def heritability(self) -> float:
        r"""Total heritability :math:`h^2 = M\pi_c\sum_h p_h\sigma_h^2`."""
        return float(self.M * self.pic * np.dot(self.weights, self.variances))

    @property
    def n_ssnp(self) -> float:
        r"""Expected number of susceptibility SNPs :math:`M\pi_c`."""
        return float(self.M * self.pic)


def resolve_model(fit_or_theta: Any, M: int | None = None) -> EffectModel:
    """Interpret a :class:`~genesis_gwas.result.FitResult` or a raw parameter vector.

    Parameters
    ----------
    fit_or_theta
        A fitted model, or a vector of 3 values ``(pic, sigmasq, a)`` (M2) or 5 values
        ``(pic, p1, sigmasq1, sigmasq2, a)`` (M3).
    M
        Number of reference SNPs. Defaults to ``fit.M`` for a
        :class:`~genesis_gwas.result.FitResult` and to :data:`DEFAULT_M` (1,070,777, the
        bundled HapMap3 reference) for a raw vector.

    Returns
    -------
    EffectModel

    Raises
    ------
    InputError
        If the vector has the wrong length or contains values outside the parameter
        space (:math:`0<\\pi_c\\le 1`, :math:`0\\le p_1\\le 1`, :math:`\\sigma_h^2>0`).
    """
    if isinstance(fit_or_theta, EffectModel):
        if M is None or int(M) == fit_or_theta.M:
            return fit_or_theta
        return EffectModel(
            n_components=fit_or_theta.n_components,
            theta=fit_or_theta.theta,
            pic=fit_or_theta.pic,
            weights=fit_or_theta.weights,
            variances=fit_or_theta.variances,
            a=fit_or_theta.a,
            M=_check_M(M),
        )
    if isinstance(fit_or_theta, FitResult):
        theta = np.asarray(fit_or_theta.estimates, dtype=np.float64).ravel()
        n_components = int(fit_or_theta.n_components)
        m_default = int(fit_or_theta.M)
    else:
        try:
            theta = np.asarray(fit_or_theta, dtype=np.float64).ravel()
        except (TypeError, ValueError) as exc:
            raise InputError(
                "Expected a FitResult or a numeric parameter vector of length 3 (M2) or 5 (M3)."
            ) from exc
        if theta.size == 3:
            n_components = 2
        elif theta.size == 5:
            n_components = 3
        else:
            raise InputError(
                f"A parameter vector must have 3 values (pic, sigmasq, a) for the two-component "
                f"model or 5 values (pic, p1, sigmasq1, sigmasq2, a) for the three-component "
                f"model; got {theta.size}."
            )
        m_default = DEFAULT_M
    spec = ModelSpec(n_components)
    theta = spec.validate(theta)
    pic, p, sig, a = spec.unpack(theta)
    if not 0.0 < pic <= 1.0:
        raise InputError(f"pic must lie in (0, 1]; got {pic}")
    if n_components == 3 and not 0.0 <= p[0] <= 1.0:
        raise InputError(f"p1 must lie in [0, 1]; got {p[0]}")
    if np.any(sig <= 0):
        raise InputError(f"Variance components must be strictly positive; got {sig.tolist()}")
    return EffectModel(
        n_components=n_components,
        theta=theta,
        pic=pic,
        weights=p,
        variances=sig,
        a=a,
        M=_check_M(M) if M is not None else m_default,
    )


def _check_M(M: Any) -> int:
    try:
        m = int(M)
    except (TypeError, ValueError) as exc:
        raise InputError("M must be a positive integer") from exc
    if m < 1 or m != M:
        raise InputError(f"M must be a positive integer; got {M!r}")
    return m


def effect_mixture(fit_or_theta: Any) -> tuple[np.ndarray, np.ndarray]:
    r"""Mixture representation of the effect-size distribution of susceptibility SNPs.

    Returns
    -------
    (weights, variances)
        Two ``(H,)`` arrays: the mixing weights :math:`p_h` (summing to one) and the
        variance components :math:`\sigma_h^2`. For M2 this is ``([1.0], [sigmasq])``;
        for M3 ``([p1, 1 - p1], [sigmasq1, sigmasq2])``.
    """
    model = resolve_model(fit_or_theta)
    return model.weights.copy(), model.variances.copy()


def effect_density(x: Any, fit_or_theta: Any) -> np.ndarray:
    r"""Probability density of the joint effect size of a susceptibility SNP.

    .. math::

        f(x) = \sum_{h=1}^{H} \frac{p_h}{\sigma_h}\,\varphi\!\Bigl(\frac{x}{\sigma_h}\Bigr)

    This is ``dmixssnp(x, est)`` of the original package (the curves of Fig. 2 in
    Zhang et al. 2018). Effects are on the standardised scale (per standard deviation of
    genotype; log odds ratio for binary traits).

    Parameters
    ----------
    x
        Effect size(s); any array shape.
    fit_or_theta
        Fitted model or parameter vector (see :func:`resolve_model`).

    Returns
    -------
    numpy.ndarray
        Density values with the shape of ``x``.
    """
    model = resolve_model(fit_or_theta)
    xv = np.asarray(x, dtype=np.float64)
    sd = model.sds
    u = xv[..., None] / sd
    dens = np.exp(-0.5 * u * u) / (_SQRT_2PI * sd)
    return np.asarray(dens @ model.weights)


def effect_cdf(x: Any, fit_or_theta: Any) -> np.ndarray:
    r"""Distribution function of the joint effect size of a susceptibility SNP.

    .. math::

        F(x) = \sum_{h=1}^{H} p_h\,\Phi\!\Bigl(\frac{x}{\sigma_h}\Bigr)

    Note that this is the distribution *among susceptibility SNPs*. The distribution
    function of a random reference SNP, which ``numInterval()`` uses internally, is
    :math:`\pi_c F(x) + (1-\pi_c)\,\mathbb 1\{x \ge 0\}`.

    Parameters
    ----------
    x
        Effect size(s); any array shape (``±inf`` allowed).
    fit_or_theta
        Fitted model or parameter vector (see :func:`resolve_model`).
    """
    model = resolve_model(fit_or_theta)
    xv = np.asarray(x, dtype=np.float64)
    return np.asarray(ndtr(xv[..., None] / model.sds) @ model.weights)


def n_in_interval(
    lower: Any,
    upper: Any,
    fit_or_theta: Any,
    M: int | None = None,
) -> np.ndarray:
    r"""Expected number of SNPs whose absolute effect size lies in ``(lower, upper)``.

    .. math::

        N(l, u) \;=\; M\,\pi_c\,\Pr\bigl(l < |\beta| < u\bigr)
                \;=\; 2\,M\,\pi_c \sum_{h=1}^{H} p_h
                      \Bigl[\Phi\!\Bigl(-\frac{l}{\sigma_h}\Bigr) - \Phi\!\Bigl(-\frac{u}{\sigma_h}\Bigr)\Bigr]

    For :math:`0 \le l < u` this equals ``numInterval(lower, upper, est, M)`` of the
    original package, which evaluates :math:`2M\{G(u)-G(l)\}` with :math:`G` the
    distribution function of a random reference SNP. The expression above is written
    with upper-tail probabilities, so it stays accurate far in the tails where the
    difference of two distribution functions cancels. A typical use is counting SNPs
    with an odds ratio above 1.01: ``n_in_interval(np.log(1.01) * sd_genotype, np.inf, fit)``.

    Parameters
    ----------
    lower, upper
        Interval limits on the absolute effect, ``0 <= lower < upper``; ``upper`` may be
        ``inf``. Arrays are broadcast against each other.
    fit_or_theta
        Fitted model or parameter vector (see :func:`resolve_model`).
    M
        Number of reference SNPs (default: that of the fit, else 1,070,777).

    Returns
    -------
    numpy.ndarray
        Expected counts, with the broadcast shape of ``lower`` and ``upper``.

    Raises
    ------
    InputError
        If a limit is negative or not finite where it must be, or ``lower >= upper``.
        (The original returns meaningless values for such inputs.)
    """
    model = resolve_model(fit_or_theta, M)
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
    if np.any(np.isnan(lo)) or np.any(np.isnan(hi)):
        raise InputError("Interval limits must not be NaN")
    if np.any(lo < 0) or not np.all(np.isfinite(lo)):
        raise InputError("lower must be finite and >= 0 (the interval is for the absolute effect)")
    lo, hi = np.broadcast_arrays(lo, hi)
    if np.any(lo >= hi):
        raise InputError("Each interval must satisfy lower < upper")
    sd = model.sds
    tail = ndtr(-lo[..., None] / sd) - ndtr(-hi[..., None] / sd)
    return np.asarray(2.0 * model.M * model.pic * (tail @ model.weights))
