r"""Sandwich variance estimation and model-selection criteria.

For a composite likelihood the naive inverse information understates uncertainty,
because the per-marker contributions are correlated through linkage disequilibrium.
Zhang et al. (2018, eq. 4) therefore use

.. math::

    \widehat{\operatorname{var}}(\hat\theta)
      = I(\hat\theta)^{-1}\, J(\hat\theta)\, I(\hat\theta)^{-1},
    \qquad
    J = \sum_{k=1}^{K} U_k \bar U_k^{\top},
    \qquad
    \bar U_k = \sum_{k' \in \mathcal N_k} U_{k'},

where :math:`U_k` is the score of marker :math:`k`, :math:`I` the total observed
information, and :math:`\mathcal N_k` the set of *GWAS markers* in the tagging
neighbourhood of marker :math:`k` (same window and :math:`r^2` threshold as the LD
score), **including** :math:`k` itself.

The same matrices give the effective number of parameters
:math:`d_s = \operatorname{tr}(I^{-1}J)` of the composite-likelihood information
criteria.

.. note::

   The original R code adds the marker's own score to :math:`\bar U_k` a *second* time
   (its tagging lists already contain the marker). ``self_double_count=True`` reproduces
   that; the default follows the published formula. See the "Differences from the
   original" page of the documentation for the measured impact.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

from genesis_gwas.exceptions import InputError, NumericalWarning

__all__ = [
    "Neighborhoods",
    "SandwichResult",
    "delta_method",
    "information_criteria",
    "neighborhood_sums",
    "sandwich",
    "scaled_inverse",
]


@dataclass(frozen=True)
class Neighborhoods:
    """Tagging neighbourhoods of the reference panel in compressed sparse row form.

    Attributes
    ----------
    indptr
        ``(n_ref + 1,)`` row pointers.
    indices
        Concatenated neighbour lists; ``indices[indptr[i]:indptr[i + 1]]`` are the
        reference rows tagged by reference SNP ``i`` (itself included).
    """

    indptr: np.ndarray
    indices: np.ndarray

    def __post_init__(self) -> None:
        indptr = np.ascontiguousarray(self.indptr, dtype=np.int64)
        indices = np.ascontiguousarray(self.indices)
        if indptr.ndim != 1 or indptr.size < 2 or indptr[0] != 0 or indptr[-1] != indices.size:
            raise InputError("Malformed CSR neighbourhood structure (indptr/indices mismatch)")
        if np.any(np.diff(indptr) < 0):
            raise InputError("Malformed CSR neighbourhood structure (indptr not monotone)")
        object.__setattr__(self, "indptr", indptr)
        object.__setattr__(self, "indices", indices)

    @property
    def n_ref(self) -> int:
        """Number of reference SNPs (rows)."""
        return int(self.indptr.size - 1)

    @classmethod
    def identity(cls, n: int) -> Neighborhoods:
        """Every SNP is its own only neighbour (independent markers)."""
        return cls(indptr=np.arange(n + 1, dtype=np.int64), indices=np.arange(n, dtype=np.int64))

    @classmethod
    def from_lists(cls, lists: list[Any]) -> Neighborhoods:
        """Build from a Python list of index sequences (convenient in tests)."""
        lens = np.fromiter((len(x) for x in lists), dtype=np.int64, count=len(lists))
        indptr = np.concatenate([[0], np.cumsum(lens)])
        indices = (
            np.concatenate([np.asarray(x, dtype=np.int64) for x in lists])
            if len(lists)
            else np.empty(0, dtype=np.int64)
        )
        return cls(indptr=indptr, indices=indices)


def neighborhood_sums(
    scores: np.ndarray,
    neighborhoods: Neighborhoods,
    ref_index: np.ndarray | None = None,
    *,
    block: int = 1 << 16,
) -> np.ndarray:
    r"""Sum the score vectors over each marker's tagging neighbourhood.

    Parameters
    ----------
    scores
        ``(K, P)`` per-marker scores.
    neighborhoods
        Tagging structure of the *reference panel*.
    ref_index
        ``(K,)`` reference row of each GWAS marker. ``None`` means the markers are
        exactly the reference SNPs, in order.
    block
        Number of markers processed per vectorised block (bounds peak memory).

    Returns
    -------
    numpy.ndarray
        ``(K, P)`` matrix whose row ``k`` is :math:`\bar U_k=\sum_{k'\in\mathcal N_k} U_{k'}`,
        the sum running over neighbours that are present among the GWAS markers.
        Reference SNPs that are tagged but absent from the GWAS contribute nothing.
    """
    S = np.asarray(scores, dtype=np.float64)
    K, P = S.shape
    n_ref = neighborhoods.n_ref
    if ref_index is None:
        if n_ref != K:
            raise InputError(
                f"ref_index is required: {K} markers but {n_ref} reference SNPs in the neighbourhoods"
            )
        ref_index = np.arange(K, dtype=np.int64)
    ref_index = np.asarray(ref_index, dtype=np.int64)
    if ref_index.shape != (K,):
        raise InputError("ref_index must have one entry per marker")
    if K and (ref_index.min() < 0 or ref_index.max() >= n_ref):
        raise InputError("ref_index contains rows outside the reference panel")
    if np.unique(ref_index).size != K:
        raise InputError("ref_index contains duplicated reference rows")

    position = np.full(n_ref, K, dtype=np.int64)  # absent reference SNPs -> zero row K
    position[ref_index] = np.arange(K, dtype=np.int64)
    S_ext = np.vstack([S, np.zeros((1, P))])
    indptr, indices = neighborhoods.indptr, neighborhoods.indices
    out = np.zeros((K, P))
    for k0 in range(0, K, block):
        k1 = min(K, k0 + block)
        rows = ref_index[k0:k1]
        starts = indptr[rows]
        lens = indptr[rows + 1] - starts
        total = int(lens.sum())
        if total == 0:
            continue
        offs = np.cumsum(lens) - lens
        flat = np.arange(total, dtype=np.int64) - np.repeat(offs, lens) + np.repeat(starts, lens)
        vals = S_ext[position[indices[flat]]]
        nonempty = lens > 0
        red = np.add.reduceat(vals, offs[nonempty], axis=0)
        out[k0:k1][nonempty] = red
    return out


@dataclass(frozen=True)
class SandwichResult:
    r"""Sandwich covariance and its ingredients.

    Attributes
    ----------
    covariance
        ``(P, P)`` covariance matrix of the parameter estimates (rows/columns of fixed
        parameters are zero).
    information
        ``(P, P)`` observed information actually used (in ``legacy`` three-component
        fits this is the asymmetric matrix of the original code).
    J
        ``(P, P)`` variability matrix :math:`\sum_k U_k \bar U_k^\top`.
    inv_information
        ``(P, P)`` inverse of the information restricted to the free parameters.
    ds
        Effective number of parameters :math:`\operatorname{tr}(I^{-1}J)`.
    information_pd
        Whether the (symmetrised) information matrix is positive definite.
    condition_number
        Condition number of the diagonally scaled information matrix.
    """

    covariance: np.ndarray
    information: np.ndarray
    J: np.ndarray
    inv_information: np.ndarray
    ds: float
    information_pd: bool
    condition_number: float

    @property
    def se(self) -> np.ndarray:
        """Standard errors (``nan`` where a variance estimate is not positive)."""
        d = np.diag(self.covariance).copy()
        with np.errstate(invalid="ignore"):
            return np.sqrt(np.where(d >= 0, d, np.nan))


def scaled_inverse(A: np.ndarray) -> tuple[np.ndarray, float]:
    r"""Invert a badly scaled square matrix via symmetric diagonal equilibration.

    The information matrix mixes parameters whose magnitudes differ by ten orders of
    magnitude (:math:`\pi_c\sim10^{-2}`, :math:`a\sim10^{-6}`), so its raw condition
    number is astronomically large although the problem is well posed. Scaling by
    ``D = diag(|A_ii|)^(-1/2)`` removes that artefact.

    Returns
    -------
    (inverse, condition_number)
        ``condition_number`` refers to the scaled matrix ``D A D``.
    """
    A = np.asarray(A, dtype=np.float64)
    d = np.abs(np.diag(A))
    if np.any(d == 0) or not np.all(np.isfinite(A)):
        return np.full_like(A, np.nan), np.inf
    D = 1.0 / np.sqrt(d)
    As = A * D[:, None] * D[None, :]
    cond = float(np.linalg.cond(As))
    if not np.isfinite(cond):
        return np.full_like(A, np.nan), np.inf
    return np.linalg.inv(As) * D[:, None] * D[None, :], cond


def sandwich(
    scores: np.ndarray,
    information: np.ndarray,
    neighborhoods: Neighborhoods | None,
    ref_index: np.ndarray | None = None,
    *,
    free: np.ndarray | None = None,
    self_double_count: bool = False,
) -> SandwichResult:
    r"""Compute the sandwich covariance :math:`I^{-1} J I^{-1}`.

    Parameters
    ----------
    scores
        ``(K, P)`` per-marker scores at the estimate.
    information
        ``(P, P)`` total observed information at the estimate.
    neighborhoods
        Tagging neighbourhoods; ``None`` treats markers as independent
        (:math:`J=\sum_k U_kU_k^\top`), which understates uncertainty for real GWAS data.
    ref_index
        Reference row of each marker (see :func:`neighborhood_sums`).
    free
        Boolean mask of estimated parameters; fixed parameters get zero variance.
    self_double_count
        Reproduce the original code's double counting of each marker's own score.
    """
    S = np.asarray(scores, dtype=np.float64)
    info = np.asarray(information, dtype=np.float64)
    P = S.shape[1]
    if info.shape != (P, P):
        raise InputError("information must be (P, P) with P = scores.shape[1]")
    free_mask = np.ones(P, dtype=bool) if free is None else np.asarray(free, dtype=bool)

    if neighborhoods is None:
        Sbar = S.copy()
    else:
        Sbar = neighborhood_sums(S, neighborhoods, ref_index)
    if self_double_count:
        Sbar = Sbar + S
    J = S.T @ Sbar

    idx = np.flatnonzero(free_mask)
    info_f = info[np.ix_(idx, idx)]
    inv_f, cond = scaled_inverse(info_f)
    sym = 0.5 * (info_f + info_f.T)
    d = np.abs(np.diag(sym))
    if np.all(d > 0) and np.all(np.isfinite(sym)):
        Dm = 1.0 / np.sqrt(d)
        eig = np.linalg.eigvalsh(sym * Dm[:, None] * Dm[None, :])
        pd = bool(np.all(eig > 0))
    else:
        pd = False
    if not pd:
        warnings.warn(
            "The observed information matrix is not positive definite at the estimate; "
            "standard errors and the model-selection criteria are unreliable. This usually "
            "means the EM algorithm has not converged or the model is over-parameterised for "
            "these data (for the three-component model: the two variance components have "
            "merged, or p1 sits on a bound).",
            NumericalWarning,
            stacklevel=2,
        )
    elif cond > 1e12:
        warnings.warn(
            f"The information matrix is nearly singular (scaled condition number {cond:.2e}); "
            "standard errors may be unstable.",
            NumericalWarning,
            stacklevel=2,
        )

    J_f = J[np.ix_(idx, idx)]
    cov_f = inv_f @ J_f @ inv_f
    cov = np.zeros((P, P))
    cov[np.ix_(idx, idx)] = cov_f
    inv_full = np.zeros((P, P))
    inv_full[np.ix_(idx, idx)] = inv_f
    ds = float(np.trace(inv_f @ J_f))
    return SandwichResult(
        covariance=cov,
        information=info,
        J=J,
        inv_information=inv_full,
        ds=ds,
        information_pd=pd,
        condition_number=cond,
    )


def information_criteria(
    loglik: float, ds: float, mean_n: float, bic_gamma: float = 0.5
) -> dict[str, float]:
    r"""Composite-likelihood information criteria used for choosing between M2 and M3.

    .. math::

        \mathrm{AIC} = -2\ell + 2 d_s, \qquad
        \mathrm{BIC} = -2\ell + d_s \log\bar n + 2\gamma\, d_s \log 5 ,

    with :math:`d_s=\operatorname{tr}(I^{-1}J)`, :math:`\bar n` the mean per-SNP sample
    size and :math:`\gamma` = ``bic_gamma`` (default 0.5). This is exactly the "modified
    BIC" of the original software (``-2*llk + ds*log(mean(n)) + 2*BIC.gamma*log(5^ds)``).
    Lower is better.
    """
    if not 0.0 <= bic_gamma <= 1.0:
        raise InputError("bic_gamma must lie in [0, 1]")
    aic = -2.0 * loglik + 2.0 * ds
    bic = -2.0 * loglik + ds * float(np.log(mean_n)) + 2.0 * bic_gamma * ds * float(np.log(5.0))
    return {"aic": float(aic), "bic": float(bic)}


def delta_method(gradient: np.ndarray, covariance: np.ndarray) -> float:
    """Standard error of a smooth function of the parameters, ``sqrt(g' V g)``."""
    g = np.asarray(gradient, dtype=np.float64)
    var = float(g @ np.asarray(covariance, dtype=np.float64) @ g)
    return float(np.sqrt(var)) if var >= 0 else float("nan")
