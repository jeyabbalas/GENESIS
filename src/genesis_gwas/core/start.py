r"""Starting values for the EM algorithm.

The scheme is the one of the original software:

1. Regress :math:`\chi^2_k = z_k^2` on :math:`x_k = \ell_k\,\bar n/M` by ordinary least
   squares (LD-score regression). Under the model
   :math:`E[\chi^2_k] = 1 + n a + n h^2 \ell_k / M`, so the slope estimates the total
   heritability :math:`h^2 = M\pi_c\sigma^2`.
2. Fix a starting proportion :math:`\pi_c^{(0)}` (default 0.01) and set
   :math:`\sigma^{2(0)} = \text{slope}/(M\pi_c^{(0)})` and
   :math:`a^{(0)} = (\text{intercept}-1)/M`.
3. For the three-component model, first fit the two-component model (with loosened
   tolerances) and start from
   :math:`(\hat\pi_c,\; 1/9,\; 5\hat\sigma^2,\; \hat\sigma^2/2,\; \hat a)`.

.. note::

   Step 2 divides the excess intercept by :math:`M`; dimensionally it should be the mean
   sample size :math:`\bar n`. Because this only affects where the EM *starts*, it is
   kept as in the original so that trajectories — and therefore early-stopped estimates
   — remain comparable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from genesis_gwas.core.data import ModelData
from genesis_gwas.exceptions import InputError

__all__ = ["LDScoreRegression", "ldsc_regression", "m2_start", "m3_start_from_m2"]

DEFAULT_STARTING_PIC = 0.01
FALLBACK_STARTING_PIC = 5e-3


@dataclass(frozen=True)
class LDScoreRegression:
    r"""Result of the unweighted LD-score regression used for initialisation.

    Attributes
    ----------
    intercept, slope
        OLS coefficients of ``chi2 ~ 1 + ldscore * mean(n) / M``. ``slope`` is a crude
        estimate of total heritability; ``intercept - 1`` measures confounding.
    mean_chi2
        Mean :math:`\chi^2` statistic.
    lambda_gc
        Genomic-control factor, ``median(chi2) / 0.4549``.
    """

    intercept: float
    slope: float
    mean_chi2: float
    lambda_gc: float

    def to_dict(self) -> dict[str, float]:
        """JSON-serialisable representation."""
        return {
            "intercept": self.intercept,
            "slope": self.slope,
            "mean_chi2": self.mean_chi2,
            "lambda_gc": self.lambda_gc,
        }


_CHI2_MEDIAN = 0.45493642311957283  # qchisq(0.5, 1)


def ldsc_regression(data: ModelData) -> LDScoreRegression:
    r"""Unweighted LD-score regression of :math:`z^2` on :math:`\ell\,\bar n/M`."""
    chisq = data.betahat**2 / data.varbetahat
    x = data.ldscore * float(np.mean(1.0 / data.varbetahat)) / data.M
    xm = float(x.mean())
    ym = float(chisq.mean())
    dx = x - xm
    sxx = float(np.sum(dx * dx))
    if sxx <= 0:
        raise InputError("LD scores are constant across SNPs; cannot derive starting values.")
    slope = float(np.sum(dx * (chisq - ym))) / sxx
    intercept = ym - slope * xm
    return LDScoreRegression(
        intercept=intercept,
        slope=slope,
        mean_chi2=ym,
        lambda_gc=float(np.median(chisq)) / _CHI2_MEDIAN,
    )


def m2_start(
    reg: LDScoreRegression,
    M: int,
    starting_pic: float | None = None,
    *,
    fallback: bool = False,
) -> np.ndarray:
    """Starting vector ``(pic, sigmasq, a)`` for the two-component model.

    Parameters
    ----------
    reg
        LD-score regression of the data.
    M
        Number of reference SNPs.
    starting_pic
        Starting proportion of susceptibility SNPs; default 0.01.
    fallback
        Use the original software's second-choice proportion (0.005) when no
        ``starting_pic`` was given; it applies this when the likelihood cannot be
        evaluated at the first choice.
    """
    if starting_pic is None:
        pic = FALLBACK_STARTING_PIC if fallback else DEFAULT_STARTING_PIC
    else:
        pic = float(starting_pic)
    if not 0.0 < pic < 1.0:
        raise InputError(f"starting_pic must lie in (0, 1); got {pic}")
    sig = reg.slope / (M * pic)
    a = (reg.intercept - 1.0) / M
    if sig < 0:
        sig, a = 1e-5, 2e-6
    return np.array([pic, sig, a])


def m3_start_from_m2(theta2: np.ndarray) -> np.ndarray:
    """Starting vector ``(pic, p1, sigmasq1, sigmasq2, a)`` from a two-component estimate.

    One ninth of the susceptibility SNPs start in a cluster with five times the
    two-component variance; the rest start with half of it.
    """
    th = np.asarray(theta2, dtype=np.float64).ravel()
    if th.size != 3:
        raise InputError("Expected a two-component estimate (pic, sigmasq, a)")
    sig1 = th[1] * 5.0
    return np.array([th[0], 1.0 / 9.0, sig1, sig1 / 10.0, th[2]])
