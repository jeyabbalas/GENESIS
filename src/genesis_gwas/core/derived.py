r"""Interpretable quantities derived from the fitted parameters, with delta-method SEs.

==========================  =========================================================
``n_ssnp``                  number of susceptibility SNPs, :math:`M\pi_c`
``heritability``            :math:`h^2 = M\pi_c\sum_h p_h\sigma_h^2` (observed scale for
                            quantitative traits, log-odds-ratio scale for binary traits)
``sibling_risk``            sibling relative risk :math:`\exp(h^2/2)` (binary traits,
                            log-odds scale)
``n_ssnp_cluster1``         M3: SNPs in the larger-variance cluster, :math:`M\pi_c p_1`
``n_ssnp_cluster2``         M3: SNPs in the smaller-variance cluster
``heritability_cluster1``   M3: :math:`M\pi_c p_1\sigma_1^2`
``heritability_cluster2``   M3: :math:`M\pi_c (1-p_1)\sigma_2^2`
==========================  =========================================================
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from genesis_gwas.core.spec import ModelSpec
from genesis_gwas.core.variance import delta_method

__all__ = ["DERIVED_LABELS", "Derived", "canonicalize_m3", "derived_quantities"]

DERIVED_LABELS: dict[str, str] = {
    "n_ssnp": "Number of susceptibility SNPs",
    "heritability": "Total heritability",
    "sibling_risk": "Sibling relative risk",
    "n_ssnp_cluster1": "Susceptibility SNPs in cluster 1 (larger variance)",
    "n_ssnp_cluster2": "Susceptibility SNPs in cluster 2 (smaller variance)",
    "heritability_cluster1": "Heritability explained by cluster 1 (larger variance)",
    "heritability_cluster2": "Heritability explained by cluster 2 (smaller variance)",
}


@dataclass(frozen=True)
class Derived:
    """A derived quantity with its delta-method standard error."""

    estimate: float
    se: float

    def to_dict(self) -> dict[str, float]:
        """JSON-serialisable representation."""
        return {"estimate": self.estimate, "se": self.se}


def derived_quantities(
    spec: ModelSpec,
    theta: np.ndarray,
    covariance: np.ndarray | None,
    M: int,
    *,
    legacy: bool = False,
) -> dict[str, Derived]:
    """Compute derived quantities and their standard errors.

    Parameters
    ----------
    spec, theta
        Model and fitted parameters.
    covariance
        Covariance matrix of ``theta`` (``None`` gives ``nan`` standard errors).
    M
        Number of reference SNPs.
    legacy
        Use the original software's (non-delta-method) formula for the standard error
        of the sibling risk, ``sqrt(risk * se_h2**2 / 2)``.
    """
    th = np.asarray(theta, dtype=np.float64)
    P = spec.n_params
    cov = (
        np.full((P, P), np.nan) if covariance is None else np.asarray(covariance, dtype=np.float64)
    )

    def se(grad: np.ndarray) -> float:
        return delta_method(grad, cov) if covariance is not None else float("nan")

    out: dict[str, Derived] = {}
    pic = th[0]
    g = np.zeros(P)
    g[0] = M
    out["n_ssnp"] = Derived(float(M * pic), se(g))

    if spec.H == 1:
        sig = th[1]
        h2 = M * pic * sig
        g = np.zeros(P)
        g[0], g[1] = M * sig, M * pic
        se_h2 = se(g)
    else:
        p1, s1, s2 = th[1], th[2], th[3]
        h2 = M * pic * (p1 * s1 + (1.0 - p1) * s2)
        g = np.array(
            [
                M * (p1 * s1 + (1.0 - p1) * s2),
                M * pic * (s1 - s2),
                M * pic * p1,
                M * pic * (1.0 - p1),
                0.0,
            ]
        )
        se_h2 = se(g)
    out["heritability"] = Derived(float(h2), se_h2)

    risk = float(np.exp(h2 / 2.0))
    if legacy:
        se_risk = float(np.sqrt(risk * se_h2**2 / 2.0))
    else:
        se_risk = 0.5 * risk * se_h2
    out["sibling_risk"] = Derived(risk, se_risk)

    if spec.H == 2:
        p1, s1, s2 = th[1], th[2], th[3]
        out["n_ssnp_cluster1"] = Derived(
            float(M * pic * p1), se(np.array([M * p1, M * pic, 0.0, 0.0, 0.0]))
        )
        out["n_ssnp_cluster2"] = Derived(
            float(M * pic * (1.0 - p1)), se(np.array([M * (1.0 - p1), -M * pic, 0.0, 0.0, 0.0]))
        )
        out["heritability_cluster1"] = Derived(
            float(M * pic * p1 * s1),
            se(np.array([M * p1 * s1, M * pic * s1, M * pic * p1, 0.0, 0.0])),
        )
        out["heritability_cluster2"] = Derived(
            float(M * pic * (1.0 - p1) * s2),
            se(np.array([M * (1.0 - p1) * s2, -M * pic * s2, 0.0, M * pic * (1.0 - p1), 0.0])),
        )
    return out


def canonicalize_m3(
    theta: np.ndarray, *matrices: np.ndarray | None
) -> tuple[np.ndarray, list[np.ndarray | None], bool]:
    r"""Relabel a three-component fit so that cluster 1 has the larger variance.

    The two non-null components are exchangeable:
    :math:`(p_1,\sigma_1^2,\sigma_2^2)\equiv(1-p_1,\sigma_2^2,\sigma_1^2)`. The paper
    defines cluster 1 as the larger-variance one. If the fit ended with
    :math:`\sigma_1^2<\sigma_2^2` the labels are swapped and any supplied ``(5, 5)``
    covariance-like matrices are transformed with the Jacobian of the relabelling.

    Returns
    -------
    (theta, matrices, swapped)
    """
    th = np.asarray(theta, dtype=np.float64).copy()
    mats: list[np.ndarray | None] = list(matrices)
    if th.size != 5 or th[2] >= th[3]:
        return th, mats, False
    A = np.zeros((5, 5))
    A[0, 0] = 1.0
    A[1, 1] = -1.0
    A[2, 3] = 1.0
    A[3, 2] = 1.0
    A[4, 4] = 1.0
    new = np.array([th[0], 1.0 - th[1], th[3], th[2], th[4]])
    out = [None if m is None else A @ np.asarray(m) @ A.T for m in mats]
    return new, out, True
