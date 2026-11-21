"""Literal, deliberately naive port of the original C++ (src/EM_EM.cpp) and R code.

This module is a *test oracle*, not part of the package. Every function mirrors its
upstream namesake line by line — scalar loops, linear-space densities, the same
truncation logic, the same (non-)normalisation, even the same typo in ``Ik3`` — so that

* it can be checked against fixtures produced by the genuine upstream code, and
* the optimised library kernels (log-space, vectorised, chunked) can be checked against
  it on small inputs.

It shares no code with ``genesis_gwas``. Upstream: yandorazhang/GENESIS @ e4e6894.
"""

from __future__ import annotations

import math

import numpy as np


# ----------------------------------------------------------------- primitives
def sumfactorial(n: int) -> float:
    """log(n!)"""
    return sum(math.log(i) for i in range(2, n + 1))


def sumfactorial_rev(n: int, k0: int) -> float:
    """log(n! / k0!)"""
    return sum(math.log(i) for i in range(k0 + 1, n + 1))


def cdnorm(x: float, mean: float, sd: float) -> float:
    res = -0.5 * math.log(2.0 * math.pi) - math.log(sd) - (x - mean) ** 2 / (2.0 * sd**2)
    return math.exp(res)


def cdbinom(k: int, size: int, prob: float) -> float:
    res = (
        sumfactorial_rev(size, size - k)
        - sumfactorial(k)
        + k * math.log(prob)
        + (size - k) * math.log(1 - prob)
    )
    return math.exp(res)


def cdmultinom3(k0: int, k1: int, k2: int, prob: tuple[float, float, float]) -> float:
    res = (
        sumfactorial_rev(k0 + k1 + k2, k2)
        - sumfactorial(k0)
        - sumfactorial(k1)
        + k0 * math.log(prob[0])
        + k1 * math.log(prob[1])
        + k2 * math.log(prob[2])
    )
    return math.exp(res)


# --------------------------------------------------------- two-component model
def weight_loglikelihood(par, betahat, varbetahat, ldscore, c0, Nstar):
    """Returns (w [K x (c0+1)], llk) exactly as upstream weight_loglikelihood()."""
    K = len(betahat)
    pi1, sigsq, a = par
    w = np.zeros((K, c0 + 1))
    res = 0.0
    for k in range(K):
        marginal = 0.0
        y = 0.0
        nk = int(Nstar[k])
        for j in range(min(nk + 1, c0 + 1)):
            tem = cdbinom(j, nk, pi1)
            tem1 = tem * cdnorm(betahat[k], 0.0, math.sqrt(j * sigsq * ldscore[k] / nk + varbetahat[k] + a))
            w[k, j] = tem1
            y += tem
            marginal += tem1
        res += math.log(marginal / y)
    w = w / w.sum(axis=1, keepdims=True)
    return w, res


def loglikelihood(par, betahat, varbetahat, ldscore, c0, Nstar):
    return weight_loglikelihood(par, betahat, varbetahat, ldscore, c0, Nstar)[1]


def update_pi1(w, Nstar):
    c0 = w.shape[1] - 1
    res1 = res2 = 0.0
    for k in range(len(Nstar)):
        nk = int(Nstar[k])
        for j in range(min(nk + 1, c0 + 1)):
            res1 += w[k, j] * j
            res2 += w[k, j] * nk
    return res1 / res2


def onestep_varcomponent(varcomponent, w, betahat, varbetahat, ldscore, Nstar):
    K, c0 = w.shape[0], w.shape[1] - 1
    sigsq, a = varcomponent
    s1 = dd1 = s2 = dd2 = dd12 = 0.0
    for k in range(K):
        nk = int(Nstar[k])
        b2 = betahat[k] ** 2
        for j in range(min(nk + 1, c0 + 1)):
            tem = (j * sigsq) * ldscore[k] / float(nk) + varbetahat[k] + a
            s1 += w[k, j] * j * ldscore[k] / (2.0 * nk) * (b2 / tem**2 - 1.0 / tem)
            s2 += w[k, j] / 2.0 * (b2 / tem**2 - 1.0 / tem)
            dd1 += w[k, j] * j * ldscore[k] * j * ldscore[k] / (2.0 * nk * nk) * (-2.0 * b2 / tem**3 + 1.0 / tem**2)
            dd2 += w[k, j] / 2.0 * (-2.0 * b2 / tem**3 + 1.0 / tem**2)
            dd12 += w[k, j] * j * ldscore[k] / (2.0 * nk) * (-2.0 * b2 / tem**3 + 1.0 / tem**2)
    det = dd1 * dd2 - dd12**2
    return np.array([sigsq - (dd2 * s1 - dd12 * s2) / det, a - (dd1 * s2 - dd12 * s1) / det])


def Sk(par, betahatk, varbetahatk, ldscorek, c0, Nstark):
    pic, sigsq, a = par
    num_pic = den = num_sig = num_a = 0.0
    for j in range(min(int(Nstark) + 1, c0 + 1)):
        tem_var = (j * sigsq) * ldscorek / Nstark + varbetahatk + a
        ww = cdbinom(j, int(Nstark), pic) * cdnorm(betahatk, 0.0, math.sqrt(tem_var))
        den += ww
        num_pic += ww * (j / pic - (Nstark - j) / (1 - pic))
        common = -0.5 / tem_var + 0.5 * betahatk**2 / tem_var**2
        num_sig += ww * common * j * ldscorek / Nstark
        num_a += ww * common
    return np.array([num_pic / den, num_sig / den, num_a / den])


def Ik(par, betahatk, varbetahatk, ldscorek, c0, Nstark):
    pic, sigsq, a = par
    Lk = 0.0
    dLk = np.zeros(3)
    ddLk = np.zeros((3, 3))
    for j in range(min(int(Nstark) + 1, c0 + 1)):
        tem_var = (j * sigsq) * ldscorek / Nstark + varbetahatk + a
        g = cdbinom(j, int(Nstark), pic) * cdnorm(betahatk, 0.0, math.sqrt(tem_var))
        common = -0.5 / tem_var + 0.5 * betahatk**2 / tem_var**2
        x = j * ldscorek / Nstark
        r = np.array([j / pic - (Nstark - j) / (1 - pic), common * x, common])
        Lk += g
        d2 = 0.5 / tem_var**2 - betahatk**2 / tem_var**3
        dr = np.zeros((3, 3))
        dr[0, 0] = -j / pic**2 - (Nstark - j) / (1 - pic) ** 2
        dr[1, 1] = d2 * x**2
        dr[1, 2] = dr[2, 1] = d2 * x
        dr[2, 2] = d2
        ddLk += g * dr + g * np.outer(r, r)
        dLk += g * r
    return np.outer(dLk, dLk) / Lk**2 - ddLk / Lk


def SS(par, betahat, varbetahat, ldscore, c0, Nstar):
    return np.array(
        [Sk(par, betahat[k], varbetahat[k], ldscore[k], c0, int(Nstar[k])) for k in range(len(betahat))]
    )


def I(par, betahat, varbetahat, ldscore, c0, Nstar):  # noqa: E743
    out = np.zeros((3, 3))
    for k in range(len(betahat)):
        out += Ik(par, betahat[k], varbetahat[k], ldscore[k], c0, int(Nstar[k]))
    return out


def EM_func(par_start, betahat, varbetahat, ldscore, Nstar, M, c0, eps1, eps2, eps3, eps, Meps, steps,
            stratification=True):
    """Returns upstream's 10-vector ``result`` (timings set to 0)."""
    par = np.array(par_start, dtype=float)
    llk = 0.0
    result = np.zeros(10)
    min_var = float(np.min(varbetahat))
    for i in range(int(Meps)):
        prev_llk = llk
        prev_par = par.copy()
        w, llk = weight_loglikelihood(par, betahat, varbetahat, ldscore, c0, Nstar)
        pi1 = update_pi1(w, Nstar)
        new_sig = np.array([par[1], par[2]])
        if not stratification:
            new_sig[1] = 0.0
        for _ in range(int(steps)):
            old_sig = onestep_varcomponent(new_sig, w, betahat, varbetahat, ldscore, Nstar)
            if not stratification:
                old_sig[1] = 0.0
            if abs(old_sig[0] - new_sig[0]) < 1e-20 and abs(old_sig[1] - new_sig[1]) < 1e-20:
                break
            new_sig = old_sig.copy()
            if new_sig[0] > 1:
                new_sig[0] = 1e-5
            if new_sig[0] < 0:
                new_sig[0] = 1e-12
            if new_sig[1] > 1:
                new_sig[1] = 1e-6
            if new_sig[1] < -min_var:
                new_sig[1] = -min_var / 2
        par = np.array([pi1, new_sig[0], new_sig[1]])
        if math.isinf(llk) or math.isnan(llk):
            par[2] = abs(par[2])
        if llk < prev_llk and c0 < 20:
            c0 = c0 + 1
        if llk < prev_llk and c0 >= 20:
            break
        result[0] = i
        result[1] = llk
        result[2:5] = par
        result[5] = par[0] * par[1] * M
        result[6] = c0
        error = np.abs(prev_par - par)
        increase_ll = (llk - prev_llk) / prev_llk if prev_llk != 0 else math.inf
        if error[0] < eps1 and error[1] < eps2 and error[2] < eps3 and abs(increase_ll) < eps:
            break
    result[1] = loglikelihood(par, betahat, varbetahat, ldscore, c0, Nstar)
    return result


# ------------------------------------------------------- three-component model
def _probs3(par):
    pic, p0 = par[0], par[1]
    pi1 = pic * p0
    pi2 = pic * (1 - p0)
    return (pi1, pi2, 1 - pi1 - pi2)


def weight_loglikelihood3(par, betahat, varbetahat, ldscore, c0, Nstar):
    K = len(betahat)
    sig1, sig2, a = par[2], par[3], par[4]
    prob = _probs3(par)
    w = np.zeros((K, (c0 + 1) * (c0 + 1)))
    res = 0.0
    for k in range(K):
        nk = int(Nstar[k])
        loginside = 0.0
        y = 0.0
        for j1 in range(min(nk + 1, c0 + 1)):
            for j2 in range(min(nk + 1, c0 + 1)):
                if nk - j1 - j2 < 0:
                    break
                tem = cdmultinom3(j1, j2, nk - j1 - j2, prob)
                tem1 = tem * cdnorm(
                    betahat[k], 0.0, math.sqrt((j1 * sig1 + j2 * sig2) * ldscore[k] / nk + varbetahat[k] + a)
                )
                w[k, j1 * (c0 + 1) + j2] = tem1
                loginside += tem1
                y += tem
        res += math.log(loginside / y)
    w = w / w.sum(axis=1, keepdims=True)
    return w, res


def loglikelihood3(par, betahat, varbetahat, ldscore, c0, Nstar):
    return weight_loglikelihood3(par, betahat, varbetahat, ldscore, c0, Nstar)[1]


def update_p3(w, Nstar):
    c0 = int(math.sqrt(w.shape[1])) - 1
    res0 = res1 = res2 = 0.0
    for k in range(len(Nstar)):
        nk = int(Nstar[k])
        for j1 in range(min(nk + 1, c0 + 1)):
            for j2 in range(min(nk + 1, c0 + 1)):
                if nk - j1 - j2 < 0:
                    break
                tem = w[k, j1 * (c0 + 1) + j2]
                res0 += tem * (j1 + j2)
                res1 += tem * nk
                res2 += tem * j1
    return np.array([res0 / res1, res2 / res0])


def onestep_varcomponent3(varcomponent, w, betahat, varbetahat, ldscore, Nstar):
    K = len(betahat)
    c0 = int(math.sqrt(w.shape[1])) - 1
    sig1, sig2, a = varcomponent
    s1 = s2 = s3 = dd11 = dd22 = dd33 = dd12 = dd13 = dd23 = 0.0
    for k in range(K):
        nk = int(Nstar[k])
        b2 = betahat[k] ** 2
        lk = ldscore[k]
        for j1 in range(min(nk + 1, c0 + 1)):
            for j2 in range(min(nk + 1, c0 + 1)):
                if nk - j1 - j2 < 0:
                    break
                wk = w[k, j1 * (c0 + 1) + j2]
                tem = (j1 * sig1 + j2 * sig2) * lk / float(nk) + varbetahat[k] + a
                first = b2 / tem**2 - 1.0 / tem
                second = -2.0 * b2 / tem**3 + 1.0 / tem**2
                s1 += wk * j1 * lk / (2.0 * nk) * first
                s2 += wk * j2 * lk / (2.0 * nk) * first
                s3 += wk / 2.0 * first
                dd11 += wk * j1 * lk * j1 * lk / (2.0 * nk * nk) * second
                dd22 += wk * j2 * lk * j2 * lk / (2.0 * nk * nk) * second
                dd33 += wk / 2.0 * second
                dd12 += wk * j1 * lk * j2 * lk / (2.0 * nk * nk) * second
                dd13 += wk * j1 * lk / (2.0 * nk) * second
                dd23 += wk * j2 * lk / (2.0 * nk) * second
    det = (
        dd11 * (dd33 * dd22 - dd23 * dd23)
        - dd12 * (dd33 * dd12 - dd23 * dd13)
        + dd13 * (dd23 * dd12 - dd22 * dd13)
    )
    r0 = sig1 - (1.0 / det) * (
        (dd33 * dd22 - dd23 * dd23) * s1 - (dd33 * dd12 - dd23 * dd13) * s2 + (dd23 * dd12 - dd22 * dd13) * s3
    )
    r1 = sig2 - (1.0 / det) * (
        -(dd33 * dd12 - dd13 * dd23) * s1 + (dd33 * dd11 - dd13 * dd13) * s2 - (dd23 * dd11 - dd12 * dd13) * s3
    )
    r2 = a - (1.0 / det) * (
        (dd23 * dd12 - dd13 * dd22) * s1 - (dd23 * dd11 - dd13 * dd12) * s2 + (dd22 * dd11 - dd12 * dd12) * s3
    )
    return np.array([r0, r1, r2])


def Sk3(par, betahatk, varbetahatk, ldscorek, c0, Nstark):
    pic, p1, sig1sq, sig2sq, a = par
    prob = (pic * p1, pic * (1 - p1), 1 - pic)
    den = 0.0
    num = np.zeros(5)
    nk = int(Nstark)
    for j1 in range(min(nk + 1, c0 + 1)):
        for j2 in range(min(nk + 1, c0 + 1)):
            if nk - j1 - j2 < 0:
                break
            tem_var = (j1 * sig1sq + j2 * sig2sq) * ldscorek / nk + varbetahatk + a
            ww = cdmultinom3(j1, j2, nk - j1 - j2, prob) * cdnorm(betahatk, 0.0, math.sqrt(tem_var))
            den += ww
            common = -0.5 / tem_var + 0.5 * betahatk**2 / tem_var**2
            num[0] += ww * ((j1 + j2) / pic - (nk - j1 - j2) / (1 - pic))
            num[1] += ww * (j1 / p1 - j2 / (1 - p1))
            num[2] += ww * common * j1 * ldscorek / nk
            num[3] += ww * common * j2 * ldscorek / nk
            num[4] += ww * common
    return num / den


def Ik3(par, betahatk, varbetahatk, ldscorek, c0, Nstark, *, reproduce_typo=True):
    """Upstream Ik3. With ``reproduce_typo`` the line ``rr(2,0) = rr(2,0);`` is honoured,
    i.e. that entry of ``rr`` stays zero, which makes the result asymmetric."""
    pic, p1, sig1sq, sig2sq, a = par
    prob = (pic * p1, pic * (1 - p1), 1 - pic)
    nk = int(Nstark)
    Lk = 0.0
    dLk = np.zeros(5)
    ddLk = np.zeros((5, 5))
    for j1 in range(min(nk + 1, c0 + 1)):
        for j2 in range(min(nk + 1, c0 + 1)):
            if nk - j1 - j2 < 0:
                break
            tem_var = (j1 * sig1sq + j2 * sig2sq) * ldscorek / nk + varbetahatk + a
            g = cdmultinom3(j1, j2, nk - j1 - j2, prob) * cdnorm(betahatk, 0.0, math.sqrt(tem_var))
            common = -0.5 / tem_var + 0.5 * betahatk**2 / tem_var**2
            x1 = j1 * ldscorek / nk
            x2 = j2 * ldscorek / nk
            r = np.array(
                [(j1 + j2) / pic - (nk - j1 - j2) / (1 - pic), j1 / p1 - j2 / (1 - p1), common * x1, common * x2, common]
            )
            Lk += g
            d2 = 0.5 / tem_var**2 - betahatk**2 / tem_var**3
            dr = np.zeros((5, 5))
            dr[0, 0] = -(j1 + j2) / pic**2 - (nk - j1 - j2) / (1 - pic) ** 2
            dr[1, 1] = -j1 / p1**2 - j2 / (1 - p1) ** 2
            dr[2, 2] = d2 * x1 * x1
            dr[2, 3] = dr[3, 2] = d2 * x1 * x2
            dr[2, 4] = dr[4, 2] = d2 * x1
            dr[3, 3] = d2 * x2 * x2
            dr[3, 4] = dr[4, 3] = d2 * x2
            dr[4, 4] = d2
            rr = np.outer(r, r)
            if reproduce_typo:
                rr[2, 0] = 0.0
            ddLk += g * dr + g * rr
            dLk += g * r
    return np.outer(dLk, dLk) / Lk**2 - ddLk / Lk


def SS3(par, betahat, varbetahat, ldscore, c0, Nstar):
    return np.array(
        [Sk3(par, betahat[k], varbetahat[k], ldscore[k], c0, int(Nstar[k])) for k in range(len(betahat))]
    )


def I3(par, betahat, varbetahat, ldscore, c0, Nstar, *, reproduce_typo=True):
    out = np.zeros((5, 5))
    for k in range(len(betahat)):
        out += Ik3(par, betahat[k], varbetahat[k], ldscore[k], c0, int(Nstar[k]), reproduce_typo=reproduce_typo)
    return out


def EM_func3(par_start, lower_pi, upper_pi, betahat, varbetahat, ldscore, Nstar, M, c0,
             eps1, eps2, eps3, eps4, eps5, eps, Meps, steps, stratification=True):
    """Returns upstream's 12-vector ``result`` (timings set to 0)."""
    par = np.array(par_start, dtype=float)
    llk = 0.0
    result = np.zeros(12)
    min_var = float(np.min(varbetahat))
    tem_p = np.array([par[0], par[1]])  # uninitialised upstream; see genesis_gwas.core.em
    for i in range(int(Meps)):
        prev_llk = llk
        prev_par = par.copy()
        pic, p0, sig1, sig2, a = par
        w, llk = weight_loglikelihood3(par, betahat, varbetahat, ldscore, c0, Nstar)
        if lower_pi[0] <= pic <= upper_pi[0] and lower_pi[1] <= p0 <= upper_pi[1]:
            tem_p = update_p3(w, Nstar)
        if pic < lower_pi[0]:
            tem_p[0] = lower_pi[0]
        if p0 < lower_pi[1]:
            tem_p[1] = lower_pi[1]
        if pic > upper_pi[0]:
            tem_p[0] = upper_pi[0]
        if p0 > upper_pi[1]:
            tem_p[1] = upper_pi[1]
        tem_sig = np.array([sig1, sig2, a])
        if not stratification:
            tem_sig[2] = 0.0
        for _ in range(int(steps)):
            old = onestep_varcomponent3(tem_sig, w, betahat, varbetahat, ldscore, Nstar)
            if not stratification:
                old[2] = 0.0
            if np.all(np.abs(old - tem_sig) < 1e-20):
                break
            tem_sig = old.copy()
        if tem_sig[0] > 1:
            tem_sig[0] = 1e-5
        if tem_sig[0] < 0:
            tem_sig[0] = 1e-12
        if tem_sig[1] > 1:
            tem_sig[1] = 1e-5
        if tem_sig[1] < 0:
            tem_sig[1] = 1e-12
        if tem_sig[2] > 1:
            tem_sig[2] = 1e-5
        if tem_sig[2] < -min_var:
            tem_sig[2] = -min_var / 2
        par = np.array([tem_p[0], tem_p[1], tem_sig[0], tem_sig[1], tem_sig[2]])
        if math.isinf(llk) or math.isnan(llk):
            par[4] = abs(par[4])
        if llk < prev_llk and c0 < 20:
            c0 = c0 + 1
        if llk < prev_llk and c0 >= 20:
            break
        result[0] = i
        result[1] = llk
        result[2:7] = par
        result[7] = M * par[0] * (par[1] * par[2] + (1 - par[1]) * par[3])
        result[8] = c0
        error = np.abs(prev_par - par)
        increase_ll = (llk - prev_llk) / prev_llk if prev_llk != 0 else math.inf
        if (error[0] < eps1 and error[1] < eps2 and error[2] < eps3 and error[3] < eps4
                and error[4] < eps5 and abs(increase_ll) < eps):
            break
    result[1] = loglikelihood3(par, betahat, varbetahat, ldscore, c0, Nstar)
    return result


# ------------------------------------------------------------ R-level variance
def sandwich_upstream(m_S, m_I, neighbor_lists):
    """genesis.R variance block: ``neighbor_lists[k]`` are the 0-based GWAS rows tagged by
    marker k (self included). Reproduces the extra ``+ m_S``."""
    m_Sbar = np.array([m_S[np.asarray(nb, dtype=int)].sum(axis=0) for nb in neighbor_lists]) + m_S
    inv_I = np.linalg.inv(m_I)
    J = m_S.T @ m_Sbar
    var_est = inv_I @ J @ inv_I
    ds = float(np.trace(inv_I @ J))
    return {"m_Sbar": m_Sbar, "inv_I": inv_I, "J": J, "var_est": var_est, "ds": ds}
