r"""Model specification: parameter layout and latent-configuration tables.

GENESIS models the *joint* effect size of each of the :math:`M` reference SNPs as

.. math::

    \beta^{(J)}_m \sim \pi_c \sum_{h=1}^{H} p_h\, N(0, \sigma_h^2) + (1-\pi_c)\,\delta_0 ,

and the *marginal* (GWAS) estimate of marker :math:`k`, which tags :math:`N^*_k`
reference SNPs with LD score :math:`\ell_k`, as a finite mixture over the latent counts
:math:`(N_k^{(1)},\dots,N_k^{(H)})` of tagged susceptibility SNPs of each type:

.. math::

    \hat\beta_k \sim \sum_{t} \Pr\nolimits_{\xi}\!\bigl(N_k = J_t\bigr)\;
    N\!\Bigl(0,\; \tfrac{\ell_k}{N_k^*}\textstyle\sum_h J_{th}\,\sigma_h^2 + s_k^2 + a\Bigr).

A *configuration* :math:`t` is one row :math:`J_t = (J_{t1},\dots,J_{tH})` of non-negative
integer counts. The sum is truncated at ``c0`` counts per component, exactly as in the
original implementation.

Two models are supported, named as in the original package:

``n_components = 2`` ("M2")
    :math:`H = 1`; parameters ``(pic, sigmasq, a)``.
``n_components = 3`` ("M3")
    :math:`H = 2`; parameters ``(pic, p1, sigmasq1, sigmasq2, a)``.

The kernels in :mod:`genesis_gwas.core.kernels` are written for general :math:`H`; this
module is the only place that knows how a flat parameter vector maps onto
:math:`(\pi_c, p, \sigma^2, a)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from scipy.special import gammaln, xlogy

from genesis_gwas.exceptions import InputError

__all__ = ["ModelSpec", "config_table", "log_multinomial_table"]


def config_table(n_effect_components: int, c0: int) -> np.ndarray:
    """Enumerate the latent configurations of the truncated mixture.

    Parameters
    ----------
    n_effect_components
        Number of non-null normal components :math:`H` (1 for M2, 2 for M3).
    c0
        Truncation level: every count satisfies ``0 <= J[t, h] <= c0``.

    Returns
    -------
    numpy.ndarray
        Integer array of shape ``((c0 + 1) ** H, H)``. Row ``t`` holds the counts
        ``(J_t1, ..., J_tH)``. For ``H = 2`` the ordering is ``t = j1 * (c0 + 1) + j2``,
        matching the column layout of the weight matrix in the original C++ code, so
        row 0 is always the all-null configuration.
    """
    if n_effect_components < 1:
        raise InputError("n_effect_components must be >= 1")
    if c0 < 1:
        raise InputError("c0 must be >= 1")
    grids = np.meshgrid(*[np.arange(c0 + 1)] * n_effect_components, indexing="ij")
    return np.stack([g.ravel() for g in grids], axis=1).astype(np.int64)


def log_multinomial_table(nstar_unique: np.ndarray, J: np.ndarray) -> np.ndarray:
    r"""Log multinomial coefficients for every (distinct :math:`N^*`, configuration) pair.

    .. math::

        \log C(N^*; J_t) = \log \frac{N^*!}{\bigl(N^* - |J_t|\bigr)!\,\prod_h J_{th}!}

    Entries with :math:`|J_t| > N^*` (more causal SNPs than tagged SNPs) are impossible
    and are set to ``-inf``; they then receive exactly zero posterior weight. This
    reproduces the ``if (Nstar - j1 - j2 < 0) break`` logic of the original code.

    Parameters
    ----------
    nstar_unique
        Distinct values of :math:`N^*_k`, shape ``(U,)``.
    J
        Configuration table from :func:`config_table`, shape ``(T, H)``.

    Returns
    -------
    numpy.ndarray
        ``float64`` array of shape ``(U, T)``.
    """
    n = np.asarray(nstar_unique, dtype=np.float64)[:, None]
    tot = J.sum(axis=1).astype(np.float64)[None, :]
    rem = n - tot
    valid = rem >= 0
    out = np.full(valid.shape, -np.inf)
    lg = (
        gammaln(n + 1.0)
        - gammaln(np.where(valid, rem, 0.0) + 1.0)
        - gammaln(J + 1.0).sum(axis=1)[None, :]
    )
    out[valid] = lg[valid]
    return out


@dataclass(frozen=True)
class ModelSpec:
    """Layout of the parameter vector for the two- or three-component model.

    Parameters
    ----------
    n_components
        ``2`` (single normal for non-null effects) or ``3`` (mixture of two normals).
        This counts the null point mass as a component, as in Zhang et al. (2018)
        and the original ``modelcomponents`` argument.
    stratification
        If ``True`` (default) the variance-inflation term ``a`` is estimated. If
        ``False`` it is fixed at zero; it stays in the parameter vector (so vectors
        always have the same length) but is excluded from :attr:`free`.
    """

    n_components: int = 2
    stratification: bool = True

    def __post_init__(self) -> None:
        if self.n_components not in (2, 3):
            raise InputError(
                f"n_components must be 2 or 3 (got {self.n_components!r}); "
                "2 = one normal component for susceptibility SNPs, 3 = a mixture of two."
            )

    # ------------------------------------------------------------------ layout
    @property
    def H(self) -> int:
        """Number of non-null (normal) effect-size components."""
        return self.n_components - 1

    @property
    def name(self) -> str:
        """Short model label, ``"M2"`` or ``"M3"``."""
        return f"M{self.n_components}"

    @cached_property
    def param_names(self) -> tuple[str, ...]:
        """Names of the entries of the parameter vector, in order."""
        if self.H == 1:
            return ("pic", "sigmasq", "a")
        return ("pic", "p1", "sigmasq1", "sigmasq2", "a")

    @property
    def n_params(self) -> int:
        """Length of the parameter vector (``a`` is always included)."""
        return len(self.param_names)

    @cached_property
    def idx_pic(self) -> int:
        r"""Index of :math:`\pi_c`."""
        return 0

    @cached_property
    def idx_p(self) -> tuple[int, ...]:
        """Indices of the free within-susceptibility mixing proportions (empty for M2)."""
        return tuple(range(1, self.H))

    @cached_property
    def idx_sig(self) -> tuple[int, ...]:
        r"""Indices of the variance components :math:`\sigma_h^2`."""
        start = self.H  # 1 (pic) + (H - 1) proportions
        return tuple(range(start, start + self.H))

    @cached_property
    def idx_a(self) -> int:
        """Index of the variance-inflation parameter :math:`a`."""
        return self.n_params - 1

    @cached_property
    def free(self) -> np.ndarray:
        """Boolean mask of estimated parameters (``a`` is fixed when ``stratification=False``)."""
        mask = np.ones(self.n_params, dtype=bool)
        if not self.stratification:
            mask[self.idx_a] = False
        return mask

    # --------------------------------------------------------------- (un)packing
    def validate(self, theta: np.ndarray) -> np.ndarray:
        """Return ``theta`` as a float64 vector of the right length, or raise."""
        th = np.asarray(theta, dtype=np.float64).ravel()
        if th.shape != (self.n_params,):
            raise InputError(
                f"{self.name} expects {self.n_params} parameters {self.param_names}, got {th.size}."
            )
        if not np.all(np.isfinite(th)):
            raise InputError(
                f"Non-finite parameter values: {dict(zip(self.param_names, th, strict=True))}"
            )
        return th

    def unpack(self, theta: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, float]:
        r"""Split a parameter vector into :math:`(\pi_c,\; p,\; \sigma^2,\; a)`.

        ``p`` has length ``H`` and sums to one (``[1.0]`` for M2; ``[p1, 1 - p1]`` for M3).
        """
        th = np.asarray(theta, dtype=np.float64)
        pic = float(th[0])
        if self.H == 1:
            p = np.array([1.0])
        else:
            p = np.array([th[1], 1.0 - th[1]])
        sig = th[list(self.idx_sig)].astype(np.float64)
        return pic, p, sig, float(th[self.idx_a])

    def pack(self, pic: float, p: np.ndarray, sig: np.ndarray, a: float) -> np.ndarray:
        """Inverse of :meth:`unpack`."""
        th = np.empty(self.n_params)
        th[0] = pic
        if self.H == 2:
            th[1] = p[0]
        th[list(self.idx_sig)] = sig
        th[self.idx_a] = a
        return th

    def log_weights(self, theta: np.ndarray, J: np.ndarray) -> tuple[np.ndarray, float]:
        r"""Parameter-dependent part of the log prior of each configuration.

        The multinomial log-probability of configuration :math:`t` for a marker tagging
        :math:`N^*` SNPs factorises as

        .. math::

            \log C(N^*; J_t)
            + \underbrace{\sum_h J_{th}\log(\pi_c p_h) - |J_t| \log(1-\pi_c)}_{b_t}
            + N^* \log(1-\pi_c) .

        Returns
        -------
        (b, log_pi0)
            ``b`` has shape ``(T,)``; ``log_pi0`` is :math:`\log(1-\pi_c)`.
        """
        pic, p, _, _ = self.unpack(theta)
        log_pi0 = float(np.log1p(-pic))
        Jf = J.astype(np.float64)
        b = xlogy(Jf, (pic * p)[None, :]).sum(axis=1) - Jf.sum(axis=1) * log_pi0
        return b, log_pi0

    def describe(self, theta: np.ndarray) -> dict[str, float]:
        """Map a parameter vector to ``{name: value}``."""
        return {
            k: float(v)
            for k, v in zip(self.param_names, np.asarray(theta, dtype=float), strict=True)
        }
