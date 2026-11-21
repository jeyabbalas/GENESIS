r"""Heritability scale conversions for binary traits.

For a case–control GWAS analysed with effective sample sizes, GENESIS estimates
heritability on the **log-odds-ratio (frailty) scale**. With :math:`P` the proportion of
cases in the study, :math:`K` the population prevalence and :math:`z` the standard
normal density at the liability threshold :math:`\Phi^{-1}(1-K)`:

.. math::

    h^2_\text{obs} = h^2_\text{log-OR}\, P(1-P), \qquad
    h^2_\text{liab} = h^2_\text{obs}\, \frac{K^2(1-K)^2}{z^2\, P(1-P)}

(Lee et al., 2011). These are the formulas of the original ``h2transfer()``.
"""

from __future__ import annotations

from scipy.stats import norm

from genesis_gwas.exceptions import InputError

__all__ = ["h2_transfer"]


def h2_transfer(
    h2_log: float,
    se_h2_log: float,
    population_prevalence: float,
    sample_prevalence: float,
) -> dict[str, float]:
    """Convert log-odds-scale heritability to the observed and liability scales.

    Parameters
    ----------
    h2_log, se_h2_log
        Heritability on the log-odds-ratio scale and its standard error.
    population_prevalence
        Lifetime prevalence of the disease in the population, in ``(0, 1)``.
    sample_prevalence
        Proportion of cases in the GWAS sample, in ``(0, 1)``.

    Returns
    -------
    dict
        ``h2_observed``, ``se_h2_observed``, ``h2_liability``, ``se_h2_liability``.
    """
    K = float(population_prevalence)
    P = float(sample_prevalence)
    if not (0.0 < K < 1.0 and 0.0 < P < 1.0):
        raise InputError("population_prevalence and sample_prevalence must lie in (0, 1)")
    z = float(norm.pdf(norm.ppf(1.0 - K)))
    scale_obs = P * (1.0 - P)
    scale_liab = (K**2) * ((1.0 - K) ** 2) / (z**2 * P * (1.0 - P))
    h2_obs = h2_log * scale_obs
    se_obs = se_h2_log * scale_obs
    return {
        "h2_observed": h2_obs,
        "se_h2_observed": se_obs,
        "h2_liability": h2_obs * scale_liab,
        "se_h2_liability": se_obs * scale_liab,
        "population_prevalence": K,
        "sample_prevalence": P,
    }
