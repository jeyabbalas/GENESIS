r"""The GENESIS composite likelihood and its derivatives.

:class:`CompositeLikelihood` binds a :class:`~genesis_gwas.core.spec.ModelSpec`, the data
(through an :class:`~genesis_gwas.core.engine.Engine`) and a truncation level ``c0``. It
turns a parameter vector into the small tables the kernels need, runs the kernels, and
assembles model-level quantities from the returned per-configuration moments.

Two log-likelihoods
-------------------
For marker :math:`k` let :math:`\pi_{kt}` be the multinomial prior of configuration
:math:`t` and :math:`f_{kt} = N(\hat\beta_k; 0, v_{kt})`. With the sum over
configurations truncated at ``c0`` counts per component:

``loglik_unnorm`` — :math:`\tilde\ell(\theta)=\sum_k \log \sum_t \pi_{kt} f_{kt}`
    The quantity whose stationary point the EM algorithm of the original software
    actually finds, and whose scores/information define the sandwich estimator.
``loglik`` — :math:`\ell(\theta)=\tilde\ell(\theta)-\sum_k \log\sum_t \pi_{kt}`
    The *renormalised* truncated likelihood that the original software reports and
    uses in its model-selection criterion.

They coincide whenever the truncated prior mass :math:`1-\sum_t\pi_{kt}` is negligible,
which is the regime the method is designed for; :meth:`CompositeLikelihood.truncation`
quantifies the gap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from genesis_gwas.core import kernels
from genesis_gwas.core.engine import Engine
from genesis_gwas.core.spec import ModelSpec, config_table, log_multinomial_table
from genesis_gwas.exceptions import InputError

__all__ = ["CompositeLikelihood", "EMPass", "Posterior", "ScoreInfo", "TruncationDiagnostics"]

_HALF_LOG_2PI = 0.5 * float(np.log(2.0 * np.pi))


def _logsumexp_rows(x: np.ndarray) -> np.ndarray:
    m = np.max(x, axis=1, keepdims=True)
    with np.errstate(divide="ignore"):
        return m[:, 0] + np.log(np.sum(np.exp(x - m), axis=1))


@dataclass(frozen=True)
class EMPass:
    """Host-level results of one pass over the data (see :meth:`CompositeLikelihood.em_pass`).

    Attributes
    ----------
    loglik
        Renormalised truncated composite log-likelihood (as reported by the original).
    loglik_unnorm
        Unnormalised truncated composite log-likelihood (the EM objective).
    counts
        ``(H,)`` expected number of tagged susceptibility SNPs of each type, summed over
        markers: :math:`A_h = \\sum_k\\sum_t w_{kt} J_{th}`.
    grad, hess
        Gradient ``(H+1,)`` and Hessian ``(H+1, H+1)`` of the expected complete-data
        log-likelihood with respect to :math:`(\\sigma_1^2,\\dots,\\sigma_H^2, a)`.
    """

    loglik: float
    loglik_unnorm: float
    counts: np.ndarray | None
    grad: np.ndarray | None
    hess: np.ndarray | None


@dataclass(frozen=True)
class ScoreInfo:
    """Per-SNP scores and the total observed information at a parameter value.

    Attributes
    ----------
    scores
        ``(K, P)`` matrix; row ``k`` is :math:`S_k = \\partial \\log \\tilde L_k/\\partial\\theta`.
    information
        ``(P, P)`` symmetric matrix :math:`-\\sum_k \\partial^2\\log\\tilde L_k/
        \\partial\\theta\\partial\\theta^\\top`.
    e_rr
        ``(P, P)`` matrix :math:`\\sum_k\\sum_t w_{kt} r_{kt} r_{kt}^\\top` (kept so that the
        asymmetric information matrix produced by a typo in the original three-component
        code can be reproduced exactly in ``legacy`` mode).
    loglik_per_snp
        ``(K,)`` contributions :math:`\\log\\tilde L_k`.
    """

    scores: np.ndarray
    information: np.ndarray
    e_rr: np.ndarray
    loglik_per_snp: np.ndarray


@dataclass(frozen=True)
class Posterior:
    """Per-SNP empirical-Bayes posterior summaries (see :meth:`CompositeLikelihood.posterior`)."""

    p_null: np.ndarray
    post_mean: np.ndarray
    post_sd: np.ndarray
    expected_count: np.ndarray


@dataclass(frozen=True)
class TruncationDiagnostics:
    """How much prior mass the truncation at ``c0`` discards.

    Attributes
    ----------
    c0
        Truncation level.
    log_mass_deficit
        :math:`-\\sum_k\\log\\sum_t\\pi_{kt} \\ge 0`: the gap between the unnormalised and
        the renormalised composite log-likelihood.
    mean_tail, max_tail
        Mean and maximum over markers of the truncated prior mass
        :math:`1-\\sum_t\\pi_{kt}`.
    frac_above_1e6
        Fraction of markers whose truncated prior mass exceeds :math:`10^{-6}`.
    """

    c0: int
    log_mass_deficit: float
    mean_tail: float
    max_tail: float
    frac_above_1e6: float

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation."""
        return {
            "c0": self.c0,
            "log_mass_deficit": self.log_mass_deficit,
            "mean_tail": self.mean_tail,
            "max_tail": self.max_tail,
            "frac_above_1e6": self.frac_above_1e6,
        }


class CompositeLikelihood:
    """Composite likelihood of one model on one data set at truncation level ``c0``.

    Parameters
    ----------
    spec
        Model specification.
    engine
        Compute engine holding the data.
    c0
        Maximum number of tagged susceptibility SNPs *per component* that is summed
        over (default 10, as in the original software).
    """

    def __init__(self, spec: ModelSpec, engine: Engine, c0: int = 10) -> None:
        self.spec = spec
        self.engine = engine
        self.c0 = 0
        self.J = np.empty((0, spec.H), dtype=np.int64)
        self.set_c0(c0)

    # ------------------------------------------------------------------ tables
    def set_c0(self, c0: int) -> None:
        """Change the truncation level and rebuild the configuration tables."""
        c0 = int(c0)
        if c0 < 1:
            raise InputError("c0 must be >= 1")
        if c0 == self.c0:
            return
        self.c0 = c0
        self.J = config_table(self.spec.H, c0)
        self._Jf = self.J.astype(np.float64)
        self._nJ = self._Jf.sum(axis=1)
        self._logC = log_multinomial_table(self.engine.nstar_unique, self.J)

    @property
    def n_terms(self) -> int:
        """Number of latent configurations ``T = (c0 + 1) ** H``."""
        return int(self.J.shape[0])

    def check_feasible(self, theta: np.ndarray) -> None:
        """Raise :class:`InputError` unless every mixture variance is positive at ``theta``."""
        pic, p, sig, a = self.spec.unpack(self.spec.validate(theta))
        if not (0.0 < pic < 1.0):
            raise InputError(f"pic must lie in (0, 1); got {pic}")
        if self.spec.H == 2 and not (0.0 < p[0] < 1.0):
            raise InputError(f"p1 must lie in (0, 1); got {p[0]}")
        if np.any(sig < 0):
            raise InputError(f"variance components must be >= 0; got {sig}")
        if a + self.engine.min_varbetahat <= 0:
            raise InputError(
                f"a = {a} makes the null-component variance non-positive "
                f"(min varbetahat = {self.engine.min_varbetahat})"
            )

    def _prior_tables(self, theta: np.ndarray) -> tuple[np.ndarray, float, float]:
        """Return ``(logprior (U, T), log_pi0, log_y_sum)`` for ``theta``."""
        b, log_pi0 = self.spec.log_weights(theta, self.J)
        logprior = self._logC + b[None, :]
        log_y_u = _logsumexp_rows(logprior)
        log_y_sum = float(np.dot(self.engine.nstar_counts, log_y_u)) + log_pi0 * self.engine.sum_nstar
        return logprior, log_pi0, log_y_sum

    # -------------------------------------------------------------------- EM pass
    def em_pass(
        self,
        theta: np.ndarray,
        eval_sig: np.ndarray | None = None,
        eval_a: float | None = None,
        *,
        want_e: bool = True,
        want_m: bool = True,
        allow_infeasible: bool = False,
    ) -> EMPass:
        r"""One pass over the data: E-step and/or derivatives of :math:`Q`.

        Parameters
        ----------
        theta
            Parameters defining the posterior configuration weights.
        eval_sig, eval_a
            Variance components at which the derivatives of
            :math:`Q(\sigma^2, a) = \sum_k\sum_t w_{kt}\log f_{kt}` are evaluated. If
            omitted they are evaluated at ``theta`` itself.
        want_e
            Compute the log-likelihoods and the expected counts.
        want_m
            Compute the gradient and Hessian of :math:`Q`.
        allow_infeasible
            Permit an evaluation point with non-positive mixture variances. Only the
            ``legacy`` EM policy uses this, to mirror the original code, which evaluates
            its (rational) Newton formulas wherever the previous step landed.
        """
        self.check_feasible(theta)
        _, _, sig, a = self.spec.unpack(theta)
        logprior, log_pi0, log_y_sum = self._prior_tables(theta)
        tables: dict[str, Any] = {
            "logprior": logprior,
            "cw": self._Jf @ sig,
            "aw": a,
            "want_e": want_e,
            "want_m": want_m,
        }
        if eval_sig is not None or eval_a is not None:
            es = sig if eval_sig is None else np.asarray(eval_sig, dtype=np.float64)
            ea = a if eval_a is None else float(eval_a)
            if not allow_infeasible and (np.any(es < 0) or ea + self.engine.min_varbetahat <= 0):
                raise InputError("Infeasible evaluation point for the variance components")
            tables["ce"] = self._Jf @ es
            tables["ae"] = ea
        sums, _ = self.engine.run(kernels.em_pass, tables, self.n_terms)

        loglik = loglik_unnorm = float("nan")
        counts = grad = hess = None
        if want_e:
            loglik_unnorm = (
                float(sums["lse"]) + log_pi0 * self.engine.sum_nstar - self.engine.K * _HALF_LOG_2PI
            )
            loglik = loglik_unnorm - log_y_sum
            counts = self._Jf.T @ sums["n_t"]
        if want_m:
            H = self.spec.H
            grad = np.empty(H + 1)
            hess = np.empty((H + 1, H + 1))
            grad[:H] = 0.5 * (self._Jf.T @ sums["a1"])
            grad[H] = 0.5 * float(np.sum(sums["a0"]))
            hess[:H, :H] = 0.5 * (self._Jf.T * sums["d2"][None, :]) @ self._Jf
            hess[:H, H] = hess[H, :H] = 0.5 * (self._Jf.T @ sums["d1"])
            hess[H, H] = 0.5 * float(np.sum(sums["d0"]))
        return EMPass(loglik=loglik, loglik_unnorm=loglik_unnorm, counts=counts, grad=grad, hess=hess)

    def loglik(self, theta: np.ndarray) -> tuple[float, float]:
        """Return ``(loglik, loglik_unnorm)`` at ``theta``."""
        res = self.em_pass(theta, want_e=True, want_m=False)
        return res.loglik, res.loglik_unnorm

    # ------------------------------------------------------------ score/information
    def score_info(self, theta: np.ndarray) -> ScoreInfo:
        """Per-SNP scores and total observed information of the truncated likelihood."""
        self.check_feasible(theta)
        spec = self.spec
        H = spec.H
        pic, p, sig, a = spec.unpack(theta)
        logprior, log_pi0, _ = self._prior_tables(theta)
        J, nJ = self._Jf, self._nJ
        ones = np.ones_like(nJ)
        if H == 1:
            Fw = nJ[:, None]
            rho = drho = None
        else:
            p1 = p[0]
            rho = J[:, 0] / p1 - J[:, 1] / (1.0 - p1)
            drho = -J[:, 0] / p1**2 - J[:, 1] / (1.0 - p1) ** 2
            Fw = np.stack([nJ, rho], axis=1)
        F1 = np.concatenate([J, ones[:, None]], axis=1)
        tables = {
            "logprior": logprior,
            "cw": J @ sig,
            "aw": a,
            "Fw": np.ascontiguousarray(Fw),
            "F1": np.ascontiguousarray(F1),
        }
        sums, per = self.engine.run(kernels.score_pass, tables, self.n_terms)

        # ---- per-SNP scores -------------------------------------------------
        P = spec.n_params
        K = self.engine.K
        nstar = self.engine.host_nstar
        r = self.engine.host_r
        alpha = 1.0 / pic + 1.0 / (1.0 - pic)
        gamma = 1.0 / (1.0 - pic)
        S = np.empty((K, P))
        S[:, 0] = alpha * per["Sw"][:, 0] - gamma * nstar
        if H == 2:
            S[:, 1] = per["Sw"][:, 1]
        for h, col in enumerate(spec.idx_sig):
            S[:, col] = r * per["S1"][:, h]
        S[:, spec.idx_a] = per["S1"][:, H]

        # ---- E_w[r r^T] and E_w[dr] summed over SNPs -------------------------
        mw_1, mw_n, mw_nn = sums["mw_1"], sums["mw_n"], sums["mw_nn"]
        m1_1, m1_n, m1_r, m1_rn = sums["m1_1"], sums["m1_n"], sums["m1_r"], sums["m1_rn"]
        m11_1, m11_r, m11_rr = sums["m11_1"], sums["m11_r"], sums["m11_rr"]
        m2_1, m2_r, m2_rr = sums["m2_1"], sums["m2_r"], sums["m2_rr"]

        e_rr = np.zeros((P, P))
        e_dr = np.zeros((P, P))
        ia = spec.idx_a
        e_rr[0, 0] = np.sum(alpha**2 * nJ**2 * mw_1 - 2 * alpha * gamma * nJ * mw_n + gamma**2 * mw_nn)
        e_dr[0, 0] = np.sum((gamma**2 - 1.0 / pic**2) * nJ * mw_1 - gamma**2 * mw_n)
        e_rr[0, ia] = np.sum(alpha * nJ * m1_1 - gamma * m1_n)
        for h, col in enumerate(spec.idx_sig):
            e_rr[0, col] = np.sum(J[:, h] * (alpha * nJ * m1_r - gamma * m1_rn))
            e_rr[col, ia] = np.sum(J[:, h] * m11_r)
            e_dr[col, ia] = np.sum(J[:, h] * m2_r)
            for h2, col2 in enumerate(spec.idx_sig):
                if col2 >= col:
                    e_rr[col, col2] = np.sum(J[:, h] * J[:, h2] * m11_rr)
                    e_dr[col, col2] = np.sum(J[:, h] * J[:, h2] * m2_rr)
        e_rr[ia, ia] = np.sum(m11_1)
        e_dr[ia, ia] = np.sum(m2_1)
        if H == 2:
            assert rho is not None and drho is not None
            e_rr[0, 1] = np.sum(rho * (alpha * nJ * mw_1 - gamma * mw_n))
            e_rr[1, 1] = np.sum(rho**2 * mw_1)
            e_dr[1, 1] = np.sum(drho * mw_1)
            e_rr[1, ia] = np.sum(rho * m1_1)
            for h, col in enumerate(spec.idx_sig):
                e_rr[1, col] = np.sum(rho * J[:, h] * m1_r)
        # symmetrise (only the upper triangle was filled)
        iu = np.triu_indices(P, 1)
        e_rr[(iu[1], iu[0])] = e_rr[iu]
        e_dr[(iu[1], iu[0])] = e_dr[iu]

        information = S.T @ S - e_rr - e_dr
        information = 0.5 * (information + information.T)
        loglik_per_snp = per["lse"] + log_pi0 * nstar - _HALF_LOG_2PI
        return ScoreInfo(scores=S, information=information, e_rr=e_rr, loglik_per_snp=loglik_per_snp)

    # -------------------------------------------------------------------- posterior
    def posterior(self, theta: np.ndarray, *, include_inflation: bool = True) -> Posterior:
        """Posterior null probability and posterior mean/SD of each marginal effect.

        Parameters
        ----------
        theta
            Fitted parameters.
        include_inflation
            If ``True`` (default) the estimated inflation ``a`` is treated as part of the
            noise variance, consistent with the likelihood. ``False`` reproduces the
            original ``postmarginal()``, which ignores ``a``.
        """
        self.check_feasible(theta)
        _, _, sig, a = self.spec.unpack(theta)
        logprior, _, _ = self._prior_tables(theta)
        tables = {
            "logprior": logprior,
            "cw": self._Jf @ sig,
            "aw": a if include_inflation else 0.0,
            "nJ": self._nJ,
        }
        _, per = self.engine.run(kernels.posterior_pass, tables, self.n_terms)
        return Posterior(
            p_null=per["p_null"],
            post_mean=per["post_mean"],
            post_sd=np.sqrt(np.maximum(per["post_var"], 0.0)),
            expected_count=per["expected_count"],
        )

    # ------------------------------------------------------------------ diagnostics
    def truncation(self, theta: np.ndarray) -> TruncationDiagnostics:
        """Quantify the prior mass discarded by truncating at ``c0`` (cheap; host only)."""
        b, log_pi0 = self.spec.log_weights(theta, self.J)
        log_y_u = _logsumexp_rows(self._logC + b[None, :]) + log_pi0 * self.engine.nstar_unique
        tail_u = np.clip(-np.expm1(np.minimum(log_y_u, 0.0)), 0.0, 1.0)
        cnt = self.engine.nstar_counts
        K = float(cnt.sum())
        return TruncationDiagnostics(
            c0=self.c0,
            log_mass_deficit=float(-np.dot(cnt, log_y_u)),
            mean_tail=float(np.dot(cnt, tail_u) / K),
            max_tail=float(tail_u.max()),
            frac_above_1e6=float(np.dot(cnt, tail_u > 1e-6) / K),
        )
