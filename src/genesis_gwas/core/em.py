r"""Expectation–maximisation for the truncated composite likelihood.

One EM iteration (identical in structure to the original software)
------------------------------------------------------------------
1. **E-step** — posterior weights :math:`w_{kt}` of every latent configuration.
2. **Proportions** — closed form:
   :math:`\pi_c = \sum_h A_h / \sum_k N^*_k` and :math:`p_h = A_h/\sum_{h'}A_{h'}`,
   where :math:`A_h=\sum_k\sum_t w_{kt}J_{th}` is the expected number of tagged
   susceptibility SNPs of type :math:`h`.
3. **Variance components** — ``newton_steps`` Newton–Raphson steps (default 2) on the
   expected complete-data log-likelihood
   :math:`Q(\sigma^2,a)=\sum_k\sum_t w_{kt}\log N(\hat\beta_k;0,v_{kt})`.

This is a generalised EM algorithm for the *unnormalised* truncated composite
likelihood :math:`\tilde\ell` (see :mod:`genesis_gwas.core.likelihood`).

Two control policies
--------------------
``legacy=False`` (default)
    The same fixed-point iteration, with the safeguards a production optimiser needs:

    * the Newton direction is replaced by a guaranteed ascent direction when the
      Hessian of :math:`Q` is not negative definite;
    * a step that would *decrease* :math:`\tilde\ell` (possible because the Newton
      steps are not an exact M-step) is halved until it does not — so the iteration is
      monotone and cannot diverge;
    * proportion bounds are imposed by projection (the exact constrained M-step);
    * with ``stratification=False`` the Newton step is taken in :math:`\sigma^2` alone;
    * the truncation level is raised (up to ``c0_max``) when the reported, renormalised
      log-likelihood decreases although :math:`\tilde\ell` increased — the signature of
      truncation at ``c0`` distorting the prior — and the iteration then *continues*
      to convergence.

    Whenever none of the safeguards is triggered, the iterates coincide with those of
    the original software to floating-point accuracy.
``legacy=True``
    A literal transcription of the control flow of the original ``EM_func`` /
    ``EM_func3`` (parameter resets, the ``c0`` rule, the early ``break``, the stale
    proportion update at the bounds). It exists so results of the original package can
    be reproduced digit for digit and is exercised against fixtures generated with the
    original C++ code.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from genesis_gwas.core.likelihood import CompositeLikelihood, EMPass
from genesis_gwas.exceptions import GenesisError, InputError

__all__ = ["EMConfig", "EMResult", "newton_direction", "run_em"]


@dataclass(frozen=True)
class EMConfig:
    """Settings of the EM algorithm (defaults are those of the original software).

    Parameters
    ----------
    c0
        Initial truncation level: at most ``c0`` tagged susceptibility SNPs per
        component are summed over.
    c0_max
        Ceiling for the adaptive increase of ``c0`` (set equal to ``c0`` to disable it).
    max_iter
        Maximum number of EM iterations.
    newton_steps
        Newton–Raphson steps per iteration for the variance components.
    tol_pic, tol_p1, tol_sigmasq, tol_a
        Absolute tolerances on the change of each parameter in one iteration.
    tol_loglik
        Tolerance on the relative change of the log-likelihood between iterations.
    pic_bounds, p1_bounds
        Box constraints on the proportions of the three-component model (the
        two-component model is unconstrained, as in the original).
    legacy
        Reproduce the control flow of the original implementation exactly.
    max_backtracks
        Maximum number of step halvings of the monotonicity safeguard.
    """

    c0: int = 10
    c0_max: int = 20
    max_iter: int = 3000
    newton_steps: int = 2
    tol_pic: float = 1e-6
    tol_p1: float = 1e-5
    tol_sigmasq: float = 1e-8
    tol_a: float = 1e-9
    tol_loglik: float = 1e-6
    pic_bounds: tuple[float, float] = (1e-7, 0.5)
    p1_bounds: tuple[float, float] = (1e-7, 0.5)
    legacy: bool = False
    max_backtracks: int = 30

    def __post_init__(self) -> None:
        if self.c0 < 1:
            raise InputError("c0 must be >= 1")
        if self.c0_max < self.c0:
            raise InputError("c0_max must be >= c0")
        if self.max_iter < 1:
            raise InputError("max_iter must be >= 1")
        if self.newton_steps < 1:
            raise InputError("newton_steps must be >= 1")
        for name in ("tol_pic", "tol_p1", "tol_sigmasq", "tol_a", "tol_loglik"):
            if not getattr(self, name) > 0:
                raise InputError(f"{name} must be > 0")
        for name in ("pic_bounds", "p1_bounds"):
            lo, hi = getattr(self, name)
            if not (0.0 < lo < hi < 1.0):
                raise InputError(f"{name} must satisfy 0 < lower < upper < 1")

    def tolerances(self, H: int) -> np.ndarray:
        """Per-parameter absolute tolerances in parameter-vector order."""
        tol = [self.tol_pic]
        if H == 2:
            tol.append(self.tol_p1)
        tol.extend([self.tol_sigmasq] * H)
        tol.append(self.tol_a)
        return np.asarray(tol)

    def loosened(self) -> EMConfig:
        """Tolerances used by the original for the M2 pre-fit that initialises M3."""
        from dataclasses import replace

        return replace(
            self,
            tol_pic=max(self.tol_pic, 1e-6),
            tol_sigmasq=max(self.tol_sigmasq, 1e-7),
            tol_a=max(self.tol_a, 1e-7),
            tol_loglik=max(self.tol_loglik, 1e-5),
        )


@dataclass
class EMResult:
    """Outcome of an EM run.

    Attributes
    ----------
    theta
        Final parameter vector.
    loglik, loglik_unnorm
        Renormalised / unnormalised truncated composite log-likelihood at ``theta``.
    c0
        Truncation level in force at the end.
    n_iter
        Number of EM iterations performed.
    converged
        Whether the convergence criteria were met.
    status
        ``"converged"``, ``"max_iter"``, ``"stalled"`` (no ascent step could be found),
        or ``"legacy_break"`` (the original's early exit at ``c0 >= 20``).
    history
        One record per iteration: ``iter``, ``loglik`` and ``loglik_unnorm`` at the
        parameters the iteration started from, ``theta`` after the update, ``c0`` and
        the number of ``backtracks``.
    n_backtracks
        Total number of step halvings performed by the monotonicity safeguard.
    """

    theta: np.ndarray
    loglik: float
    loglik_unnorm: float
    c0: int
    n_iter: int
    converged: bool
    status: str
    history: list[dict[str, Any]] = field(default_factory=list)
    n_backtracks: int = 0


Callback = Callable[[dict[str, Any]], None]


def newton_direction(grad: np.ndarray, hess: np.ndarray) -> tuple[np.ndarray, bool]:
    r"""Ascent direction for maximising a function with gradient ``grad`` and Hessian ``hess``.

    If ``hess`` is negative definite this is the Newton step :math:`-H^{-1}g`.
    Otherwise the eigenvalues of the (diagonally scaled) Hessian are replaced by minus
    their absolute values, which yields a direction of guaranteed ascent.

    Returns
    -------
    (direction, modified)
        ``modified`` is ``True`` when the plain Newton step could not be used.
    """
    g = np.asarray(grad, dtype=np.float64)
    Hm = np.asarray(hess, dtype=np.float64)
    if g.size == 0:
        return g.copy(), False
    if not (np.all(np.isfinite(g)) and np.all(np.isfinite(Hm))):
        return np.zeros_like(g), True
    diag = np.abs(np.diag(Hm))
    scale = np.where(diag > 0, 1.0 / np.sqrt(np.where(diag > 0, diag, 1.0)), 1.0)
    Hs = Hm * scale[:, None] * scale[None, :]
    evals, evecs = np.linalg.eigh(0.5 * (Hs + Hs.T))
    top = float(np.max(np.abs(evals))) if evals.size else 0.0
    if top == 0.0:
        return np.zeros_like(g), True
    if np.all(evals < -1e-12 * top):
        try:
            return -np.linalg.solve(Hm, g), False
        except np.linalg.LinAlgError:  # pragma: no cover - guarded by the eigen test
            pass
    lam = -np.maximum(np.abs(evals), 1e-8 * top)
    gs = scale * g
    return -scale * (evecs @ ((evecs.T @ gs) / lam)), True


def run_em(
    lik: CompositeLikelihood,
    theta0: np.ndarray,
    config: EMConfig | None = None,
    callback: Callback | None = None,
) -> EMResult:
    """Maximise the composite likelihood by EM, starting from ``theta0``.

    Parameters
    ----------
    lik
        Likelihood object (model, data, engine). Its truncation level is set to
        ``config.c0`` and may be raised during the run.
    theta0
        Starting values in the layout of ``lik.spec.param_names``.
    config
        Algorithm settings.
    callback
        Called once per iteration with that iteration's history record.
    """
    cfg = config or EMConfig()
    theta = lik.spec.validate(theta0).copy()
    lik.set_c0(cfg.c0)
    if cfg.legacy:
        return _run_legacy(lik, theta, cfg, callback)
    return _run_safeguarded(lik, theta, cfg, callback)


# ---------------------------------------------------------------------------
# Default policy
# ---------------------------------------------------------------------------
def _prepare_start(lik: CompositeLikelihood, theta: np.ndarray, cfg: EMConfig) -> np.ndarray:
    spec = lik.spec
    pic, p, sig, a = spec.unpack(theta)
    if not spec.stratification:
        a = 0.0
    if spec.H == 2:
        pic = float(np.clip(pic, *cfg.pic_bounds))
        p = np.array([np.clip(p[0], *cfg.p1_bounds), 0.0])
        p[1] = 1.0 - p[0]
    start = spec.pack(pic, p, sig, a)
    lik.check_feasible(start)
    return start


def _mstep(
    lik: CompositeLikelihood, theta: np.ndarray, res: EMPass, cfg: EMConfig
) -> tuple[np.ndarray, int]:
    """One (generalised) M-step. Returns the new parameters and the number of modified Newton steps."""
    spec = lik.spec
    H = spec.H
    _, _, sig, a = spec.unpack(theta)
    assert res.counts is not None and res.grad is not None and res.hess is not None

    total = float(res.counts.sum())
    pic_new = total / lik.engine.sum_nstar
    if H == 2:
        pic_new = float(np.clip(pic_new, *cfg.pic_bounds))
        p1 = float(res.counts[0] / total) if total > 0 else cfg.p1_bounds[0]
        p1 = float(np.clip(p1, *cfg.p1_bounds))
        p_new = np.array([p1, 1.0 - p1])
    else:
        pic_new = float(np.clip(pic_new, 1e-300, 1.0 - 1e-12))
        p_new = np.array([1.0])

    free = np.ones(H + 1, dtype=bool)
    free[H] = spec.stratification
    phi = np.append(sig, a if spec.stratification else 0.0)
    grad, hess = res.grad, res.hess
    min_s2 = lik.engine.min_varbetahat
    n_modified = 0
    for step in range(cfg.newton_steps):
        direction, modified = newton_direction(grad[free], hess[np.ix_(free, free)])
        n_modified += int(modified)
        new = phi.copy()
        new[free] += direction
        neg = new[:H] < 0
        new[:H][neg] = 1e-12
        if spec.stratification and new[H] <= -min_s2:
            new[H] = -min_s2 / 2.0
        if np.all(np.abs(new - phi) < 1e-20):
            break
        phi = new
        if step + 1 < cfg.newton_steps:
            more = lik.em_pass(theta, eval_sig=phi[:H], eval_a=float(phi[H]), want_e=False)
            assert more.grad is not None and more.hess is not None
            grad, hess = more.grad, more.hess
    return spec.pack(pic_new, p_new, phi[:H], float(phi[H])), n_modified


def _run_safeguarded(
    lik: CompositeLikelihood, theta: np.ndarray, cfg: EMConfig, callback: Callback | None
) -> EMResult:
    spec = lik.spec
    tol = cfg.tolerances(spec.H)
    theta = _prepare_start(lik, theta, cfg)
    res = lik.em_pass(theta)
    if not np.isfinite(res.loglik_unnorm):
        raise GenesisError(
            "The composite log-likelihood is not finite at the starting values "
            f"{spec.describe(theta)}; choose different starting values."
        )
    history: list[dict[str, Any]] = []
    prev_loglik: float | None = None
    status = "max_iter"
    total_backtracks = 0
    n_iter = 0

    def advance(
        current: np.ndarray, current_res: EMPass, proposal: np.ndarray
    ) -> tuple[np.ndarray, EMPass, int, bool]:
        """E-step at ``proposal``, halving the step while the EM objective decreases."""
        slack = 1e-10 * max(1.0, abs(current_res.loglik_unnorm))
        step = proposal - current
        scale = 1.0
        for nb in range(cfg.max_backtracks + 1):
            cand = current + scale * step
            cand_res = lik.em_pass(cand)
            if np.isfinite(cand_res.loglik_unnorm) and (
                cand_res.loglik_unnorm >= current_res.loglik_unnorm - slack
            ):
                return cand, cand_res, nb, True
            scale *= 0.5
        return current, current_res, cfg.max_backtracks, False

    for it in range(cfg.max_iter):
        n_iter = it + 1
        proposal, n_modified = _mstep(lik, theta, res, cfg)

        raise_c0 = prev_loglik is not None and res.loglik < prev_loglik and lik.c0 < cfg.c0_max
        delta = np.abs(proposal - theta)
        rel_ll = (
            abs((res.loglik - prev_loglik) / prev_loglik)
            if prev_loglik not in (None, 0.0)
            else np.inf
        )
        converged = bool(np.all(delta < tol) and rel_ll < cfg.tol_loglik)
        if raise_c0:
            lik.set_c0(lik.c0 + 1)

        new_theta, new_res, nb, ok = advance(theta, res, proposal)
        total_backtracks += nb
        record = {
            "iter": it,
            "loglik": res.loglik,
            "loglik_unnorm": res.loglik_unnorm,
            "theta": new_theta.copy(),
            "c0": lik.c0,
            "backtracks": nb,
            "modified_newton": n_modified,
        }
        history.append(record)
        if callback is not None:
            callback(record)
        if not ok:
            status = "stalled"
            break
        prev_loglik = res.loglik
        theta, res = new_theta, new_res
        if converged and nb == 0:
            status = "converged"
            break

    return EMResult(
        theta=theta,
        loglik=res.loglik,
        loglik_unnorm=res.loglik_unnorm,
        c0=lik.c0,
        n_iter=n_iter,
        converged=status == "converged",
        status=status,
        history=history,
        n_backtracks=total_backtracks,
    )


# ---------------------------------------------------------------------------
# Legacy policy: literal port of EM_func / EM_func3
# ---------------------------------------------------------------------------
def _legacy_newton(grad: np.ndarray, hess: np.ndarray) -> np.ndarray:
    """Raw Newton increment ``-H^{-1} g`` (no safeguards), as in ``onestep_varcomponent*``."""
    try:
        return -np.linalg.solve(hess, grad)
    except np.linalg.LinAlgError:
        return np.full_like(grad, np.nan)


def _run_legacy(
    lik: CompositeLikelihood, theta: np.ndarray, cfg: EMConfig, callback: Callback | None
) -> EMResult:
    spec = lik.spec
    H = spec.H
    tol = cfg.tolerances(H)
    min_s2 = lik.engine.min_varbetahat
    sum_nstar = lik.engine.sum_nstar
    strat = spec.stratification
    legacy_c0_cap = 20  # hard-coded in the original

    par = theta.copy()
    llk = 0.0
    c0 = lik.c0
    result_par = par.copy()
    result_c0 = c0
    status = "max_iter"
    history: list[dict[str, Any]] = []
    n_iter = 0
    # The original leaves the stale proportion vector uninitialised; starting it at the
    # current values is the only sane reading and is what happens whenever the start is
    # inside the bounds.
    stale_p = np.array([par[0], par[1]]) if H == 2 else None

    for it in range(cfg.max_iter):
        n_iter = it + 1
        prev_llk = llk
        prev_par = par.copy()
        pic, p, sig, a = spec.unpack(par)

        try:
            res = lik.em_pass(par)
        except InputError as exc:
            raise GenesisError(
                "legacy mode: the original algorithm reached parameter values at which its "
                f"likelihood is undefined ({exc}); the original would return NaN here. "
                "Re-run without legacy=True."
            ) from exc
        llk = res.loglik
        assert res.counts is not None and res.grad is not None and res.hess is not None

        # -- proportions
        total = float(res.counts.sum())
        if H == 1:
            new_pic = total / sum_nstar
            new_p1 = None
        else:
            assert stale_p is not None
            lo0, up0 = cfg.pic_bounds
            lo1, up1 = cfg.p1_bounds
            p0 = float(p[0])
            if lo0 <= pic <= up0 and lo1 <= p0 <= up1:
                stale_p = np.array([total / sum_nstar, res.counts[0] / total])
            if pic < lo0:
                stale_p[0] = lo0
            if p0 < lo1:
                stale_p[1] = lo1
            if pic > up0:
                stale_p[0] = up0
            if p0 > up1:
                stale_p[1] = up1
            new_pic, new_p1 = float(stale_p[0]), float(stale_p[1])

        # -- variance components: raw joint Newton steps on (sigma^2..., a)
        cur = np.append(sig, a)
        if not strat:
            cur[H] = 0.0
        grad, hess = res.grad, res.hess
        at_weights_point = bool(strat or a == 0.0)
        for _step in range(cfg.newton_steps):
            if not at_weights_point:
                more = lik.em_pass(
                    par, eval_sig=cur[:H], eval_a=float(cur[H]), want_e=False, allow_infeasible=True
                )
                assert more.grad is not None and more.hess is not None
                grad, hess = more.grad, more.hess
            old = cur + _legacy_newton(grad, hess)
            if not strat:
                old[H] = 0.0
            if np.all(np.abs(old - cur) < 1e-20):
                break
            cur = old
            at_weights_point = False
            if H == 1:  # the two-component code clamps inside the loop
                if cur[0] > 1:
                    cur[0] = 1e-5
                if cur[0] < 0:
                    cur[0] = 1e-12
                if cur[1] > 1:
                    cur[1] = 1e-6
                if cur[1] < -min_s2:
                    cur[1] = -min_s2 / 2.0
        if H == 2:  # the three-component code clamps once, after the loop
            for h in range(2):
                if cur[h] > 1:
                    cur[h] = 1e-5
                if cur[h] < 0:
                    cur[h] = 1e-12
            if cur[2] > 1:
                cur[2] = 1e-5
            if cur[2] < -min_s2:
                cur[2] = -min_s2 / 2.0
        if not np.all(np.isfinite(cur)):
            raise GenesisError(
                "legacy mode: the unsafeguarded Newton step of the original algorithm produced "
                "non-finite variance components (the original would return NaN). "
                "Re-run without legacy=True."
            )

        p_new = np.array([1.0]) if H == 1 else np.array([new_p1, 1.0 - float(new_p1)])  # type: ignore[arg-type]
        par = spec.pack(new_pic, p_new, cur[:H], float(cur[H]))

        if not np.isfinite(llk):
            par[spec.idx_a] = abs(par[spec.idx_a])
        decreased = llk < prev_llk
        if decreased and c0 < legacy_c0_cap:
            c0 += 1
            lik.set_c0(c0)
        if decreased and c0 >= legacy_c0_cap:
            status = "legacy_break"
            break

        result_par = par.copy()
        result_c0 = c0
        record = {
            "iter": it,
            "loglik": llk,
            "loglik_unnorm": res.loglik_unnorm,
            "theta": par.copy(),
            "c0": c0,
            "backtracks": 0,
            "modified_newton": 0,
        }
        history.append(record)
        if callback is not None:
            callback(record)

        err = np.abs(prev_par - par)
        with np.errstate(divide="ignore", invalid="ignore"):
            increase = (llk - prev_llk) / prev_llk if prev_llk != 0 else np.inf
        if np.all(err < tol) and abs(increase) < cfg.tol_loglik:
            status = "converged"
            break

    # Final log-likelihood at the latest parameters and truncation level, exactly as the
    # original does; on an early break the *reported* parameters are one iteration older.
    try:
        final = lik.em_pass(par, want_m=False)
        final_ll, final_llu = final.loglik, final.loglik_unnorm
    except InputError:
        final_ll = final_llu = float("nan")
    lik.set_c0(result_c0)
    return EMResult(
        theta=result_par,
        loglik=final_ll,
        loglik_unnorm=final_llu,
        c0=result_c0,
        n_iter=n_iter,
        converged=status == "converged",
        status=status,
        history=history,
        n_backtracks=0,
    )
