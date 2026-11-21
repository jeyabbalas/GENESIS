r"""Backend-agnostic numerical kernels (the inner loops of GENESIS).

Every kernel processes one *chunk* of SNPs and is written against an array namespace
``xp`` that is either :mod:`numpy` or :mod:`torch`; only operations that exist with
identical names and semantics in both are used. A kernel returns two dictionaries:

``sums``
    Small arrays that are *added* across chunks and devices (map-reduce).
``per_snp``
    Arrays with one row per SNP of the chunk; these are concatenated in SNP order.

Notation (one chunk of ``n`` SNPs, ``T`` latent configurations)
----------------------------------------------------------------
``b``, ``b2``   :math:`\hat\beta_k`, :math:`\hat\beta_k^2`
``s2``          :math:`s_k^2` (sampling variance of :math:`\hat\beta_k`)
``r``           :math:`\ell_k / N^*_k` (LD score per tagged SNP)
``uidx``        index of :math:`N^*_k` in the table of distinct :math:`N^*` values
``logprior``    ``(U, T)`` log prior of each configuration, without the term
                :math:`N^*_k\log(1-\pi_c)` that is constant across configurations
``c``           ``(T,)`` :math:`\sum_h J_{th}\sigma_h^2`
``v``           ``(n, T)`` mixture variances :math:`v_{kt} = r_k c_t + s_k^2 + a`

Working in log space throughout (log-sum-exp for the mixture, never a raw density) is
what makes the computation immune to the underflow that affects the original
linear-space implementation on platforms where ``long double`` is 64-bit.
"""

from __future__ import annotations

from typing import Any

__all__ = ["Chunk", "em_pass", "posterior_pass", "score_pass"]


class Chunk:
    """A contiguous block of SNPs living on one device (thin attribute container)."""

    __slots__ = ("b", "b2", "nstar", "r", "r2", "s2", "uidx")

    def __init__(self, b: Any, b2: Any, s2: Any, r: Any, r2: Any, nstar: Any, uidx: Any) -> None:
        self.b = b
        self.b2 = b2
        self.s2 = s2
        self.r = r
        self.r2 = r2
        self.nstar = nstar
        self.uidx = uidx


def _estep(xp: Any, ch: Chunk, logprior: Any, c: Any, a: float) -> tuple[Any, Any, Any, Any]:
    r"""Posterior configuration weights for one chunk.

    Returns
    -------
    w : ``(n, T)``
        Posterior weights :math:`w_{kt}` (rows sum to one; impossible configurations
        have weight exactly zero).
    q : ``(n, T)``
        :math:`1/v_{kt}`.
    u : ``(n, T)``
        :math:`\hat\beta_k^2 / v_{kt}`.
    lse : ``(n,)``
        :math:`\log\sum_t \exp(\text{logprior}_{kt} - \tfrac12\log v_{kt} - \tfrac12 u_{kt})`,
        i.e. the per-SNP log-likelihood up to the additive constants
        :math:`N^*_k\log(1-\pi_c) - \tfrac12\log 2\pi`.
    """
    v = ch.r[:, None] * c[None, :]
    v += (ch.s2 + a)[:, None]
    q = 1.0 / v
    u = ch.b2[:, None] * q
    lj = xp.log(v)
    lj += u
    lj *= -0.5
    lj += logprior[ch.uidx]
    m = xp.amax(lj, axis=1, keepdims=True)
    lj -= m
    w = xp.exp(lj, out=lj)
    se = xp.sum(w, axis=1, keepdims=True)
    w /= se
    lse = m[:, 0] + xp.log(se[:, 0])
    return w, q, u, lse


def em_pass(xp: Any, ch: Chunk, tb: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    r"""Fused E-step and M-step sufficient statistics.

    Tables ``tb``
    -------------
    ``logprior``, ``cw``, ``aw``
        Define the posterior weights (the "old" parameters of the EM iteration).
    ``ce``, ``ae`` (optional)
        Variance components at which the derivatives of the expected complete-data
        log-likelihood :math:`Q` are evaluated. When absent, the derivatives are taken
        at the weight parameters themselves, re-using every intermediate (this is the
        first Newton step of each EM iteration, fused with the E-step).
    ``want_e``, ``want_m``
        Which groups of statistics to return.

    Returned sums (all of shape ``(T,)`` unless noted)
    -------------------------------------------------
    ``lse``  (scalar)  :math:`\sum_k \mathrm{lse}_k`
    ``n_t``            :math:`\sum_k w_{kt}` — expected number of SNPs in configuration ``t``
    ``a0``, ``a1``     :math:`\sum_k \{1, r_k\}\, w_{kt}\, q_{kt}(u_{kt}-1)`
    ``d0``, ``d1``, ``d2``
                       :math:`\sum_k \{1, r_k, r_k^2\}\, w_{kt}\, q_{kt}^2 (1-2u_{kt})`

    With :math:`\partial_v \log f = \tfrac12 q(u-1)` and
    :math:`\partial_v^2 \log f = \tfrac12 q^2(1-2u)`, the host assembles the gradient and
    Hessian of :math:`Q` with respect to :math:`(\sigma_1^2,\dots,\sigma_H^2, a)` from
    these vectors and the configuration table (the factor ½ is applied there).
    """
    w, q, u, lse = _estep(xp, ch, tb["logprior"], tb["cw"], tb["aw"])
    sums: dict[str, Any] = {}
    if tb["want_e"]:
        sums["lse"] = xp.sum(lse)
        sums["n_t"] = xp.sum(w, axis=0)
    if tb["want_m"]:
        ce = tb.get("ce")
        if ce is not None:
            v = ch.r[:, None] * ce[None, :]
            v += (ch.s2 + tb["ae"])[:, None]
            q = 1.0 / v
            u = ch.b2[:, None] * q
        g = u - 1.0
        g *= q
        g *= w
        sums["a0"] = xp.sum(g, axis=0)
        sums["a1"] = xp.einsum("k,kt->t", ch.r, g)
        h = u * (-2.0)
        h += 1.0
        h *= q
        h *= q
        h *= w
        sums["d0"] = xp.sum(h, axis=0)
        sums["d1"] = xp.einsum("k,kt->t", ch.r, h)
        sums["d2"] = xp.einsum("k,kt->t", ch.r2, h)
    return sums, {}


def score_pass(xp: Any, ch: Chunk, tb: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    r"""Per-SNP score ingredients and the moments needed for the information matrix.

    For configuration ``t`` of SNP ``k`` let :math:`r_{kt} = \partial\log(\text{joint}_{kt})
    /\partial\theta`. The per-SNP score is :math:`S_k = \sum_t w_{kt} r_{kt}` and the
    observed information is
    :math:`I_k = S_k S_k^\top - \sum_t w_{kt}\,(r_{kt} r_{kt}^\top + \partial r_{kt}/\partial\theta^\top)`.

    Every entry of :math:`r_{kt}` is a product of a function of the configuration alone
    (counts, :math:`\rho_t`), a function of the SNP alone (:math:`1`, :math:`N^*_k`,
    :math:`r_k`), and one of ``1`` or :math:`g_1 = \partial_v \log f`. The kernel therefore
    only has to return

    * per-SNP contractions ``Sw = w @ Fw`` and ``S1 = (w * g1) @ F1`` with the small
      configuration-feature matrices ``Fw`` and ``F1`` supplied by the host, and
    * thirteen ``(T,)`` moment vectors :math:`\sum_k c_k X_{kt}` for
      :math:`X \in \{w,\; w g_1,\; w g_1^2,\; w g_2\}` and SNP-level factors :math:`c_k`.

    The host (:class:`genesis_gwas.core.likelihood.CompositeLikelihood`) assembles the
    model-specific score matrix and information matrix from these.
    """
    w, q, u, lse = _estep(xp, ch, tb["logprior"], tb["cw"], tb["aw"])
    g1 = u - 1.0
    g1 *= q
    g1 *= 0.5
    w1 = w * g1
    w11 = w1 * g1
    g2 = u * (-2.0)
    g2 += 1.0
    g2 *= q
    g2 *= q
    g2 *= 0.5
    g2 *= w  # now w * g2

    nst = ch.nstar
    rn = ch.r * nst
    sums = {
        "lse": xp.sum(lse),
        "mw_1": xp.sum(w, axis=0),
        "mw_n": xp.einsum("k,kt->t", nst, w),
        "mw_nn": xp.einsum("k,kt->t", nst * nst, w),
        "m1_1": xp.sum(w1, axis=0),
        "m1_n": xp.einsum("k,kt->t", nst, w1),
        "m1_r": xp.einsum("k,kt->t", ch.r, w1),
        "m1_rn": xp.einsum("k,kt->t", rn, w1),
        "m11_1": xp.sum(w11, axis=0),
        "m11_r": xp.einsum("k,kt->t", ch.r, w11),
        "m11_rr": xp.einsum("k,kt->t", ch.r2, w11),
        "m2_1": xp.sum(g2, axis=0),
        "m2_r": xp.einsum("k,kt->t", ch.r, g2),
        "m2_rr": xp.einsum("k,kt->t", ch.r2, g2),
    }
    per_snp = {
        "Sw": xp.einsum("kt,tq->kq", w, tb["Fw"]),
        "S1": xp.einsum("kt,tq->kq", w1, tb["F1"]),
        "lse": lse,
    }
    return sums, per_snp


def posterior_pass(xp: Any, ch: Chunk, tb: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    r"""Empirical-Bayes posterior summaries of each marker's marginal effect.

    Conditional on configuration ``t`` the marginal effect has prior
    :math:`N(0, \delta_{kt})` with :math:`\delta_{kt} = r_k c_t`, and
    :math:`\hat\beta_k \mid \beta_k \sim N(\beta_k, s_k^2 + a)`. Hence

    .. math::

        E[\beta_k \mid \hat\beta_k] = \hat\beta_k \sum_t w_{kt}\,
            \frac{\delta_{kt}}{\delta_{kt} + s_k^2 + a},
        \qquad
        \Pr(\beta_k = 0 \mid \hat\beta_k) = w_{k0}.

    ``tb["aw"]`` is the noise inflation used both for the weights and the shrinkage
    (the fitted ``a``; pass 0 to reproduce the original ``postmarginal``).
    """
    c = tb["cw"]
    a = tb["aw"]
    w, q, _u, lse = _estep(xp, ch, tb["logprior"], c, a)
    shrink = ch.r[:, None] * c[None, :]
    shrink *= q
    ws = w * shrink
    factor = xp.sum(ws, axis=1)
    ws *= shrink
    factor2 = xp.sum(ws, axis=1)
    post_mean = ch.b * factor
    second = ch.b2 * factor2 + (ch.s2 + a) * factor
    per_snp = {
        "p_null": w[:, 0],
        "post_mean": post_mean,
        "post_var": second - post_mean * post_mean,
        "expected_count": xp.einsum("kt,t->k", w, tb["nJ"]),
        "lse": lse,
    }
    return {}, per_snp
