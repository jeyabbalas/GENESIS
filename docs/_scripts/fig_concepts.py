"""Figures for the "model in pictures" chapter.

* ``ld-neighbourhood.svg``   – a real tagging neighbourhood (N*, LD score)
* ``joint-vs-marginal.svg``  – how LD turns one joint effect into many marginal effects
* ``effect-prior.svg``       – the spike-and-normal(s) effect-size distribution (eq. 1)
* ``marginal-mixture.svg``   – the mixture distribution of a marker's z statistic (eq. 3)
"""

from __future__ import annotations

import numpy as np
from _common import GREY, LIGHT, PALETTE, base_parser, load_fit, load_reference, new_figure, panel_label, save
from scipy.stats import binom, norm

INDEX_SNP = "rs724016"


def neighbourhood(ref, snp: str):
    """Positions (Mb), r^2 and tagging status of all reference SNPs within the window."""
    k = int(ref.lookup([snp])[0])
    if k < 0:
        raise SystemExit(f"{snp} is not in reference {ref.name}")
    nb = ref.neighborhoods
    lo, hi = int(nb.indptr[k]), int(nb.indptr[k + 1])
    tagged = np.asarray(nb.indices[lo:hi])
    r2 = np.asarray(ref.pair_r2()[lo:hi], dtype=float)
    window = ref.config.window_kb * 1e3
    pos = ref.pos.astype(np.int64)
    in_window = np.flatnonzero((ref.chrom == ref.chrom[k]) & (np.abs(pos - pos[k]) <= window))
    untagged = np.setdiff1d(in_window, tagged)
    return k, tagged, r2, untagged


def fig_ld_neighbourhood(ref, out) -> None:
    k, tagged, r2, untagged = neighbourhood(ref, INDEX_SNP)
    pos = ref.pos.astype(float) / 1e6
    n_ref = ref.n_ref
    adj = r2 - (1.0 - r2) / (n_ref - 2)
    fig, ax = new_figure(figsize=(6.8, 3.0))
    ax.axhline(ref.config.r2_min, color=GREY, lw=0.8, ls="--")
    ax.vlines(pos[tagged], 0, r2, color=PALETTE[1], lw=0.9, alpha=0.8)
    ax.plot(pos[tagged], r2, "o", ms=3.2, color=PALETTE[1], label=f"tagged: $r^2 \\geq {ref.config.r2_min:g}$ (n = {tagged.size})")
    ax.plot(pos[untagged], np.full(untagged.size, -0.035), "|", ms=5, color=GREY, alpha=0.7,
            label=f"in the window, not tagged (n = {untagged.size})")
    ax.plot([pos[k]], [1.0], "D", ms=6, color=PALETTE[2], label=f"marker {INDEX_SNP}", zorder=5)
    half = ref.config.window_kb / 1e3
    ax.set_xlim(pos[k] - half * 1.02, pos[k] + half * 1.02)
    ax.set_ylim(-0.07, 1.08)
    ax.set_xlabel(f"Position on chromosome {int(ref.chrom[k])} (Mb, {ref.genome_build})")
    ax.set_ylabel(f"$r^2$ with {INDEX_SNP}")
    ax.text(pos[k] + half * 1.0, ref.config.r2_min + 0.015, "tagging threshold", ha="right", va="bottom", color=GREY, fontsize=8)
    ax.text(
        0.015, 0.97,
        f"$N^*_k$ = {int(ref.nstar[k])} tagged SNPs\n$\\ell_k$ = {ref.ldscore[k]:.1f}  (sum of bias-corrected $r^2$)",
        transform=ax.transAxes, va="top", ha="left", fontsize=9,
    )
    ax.legend(loc="upper right")
    assert abs(adj.sum() - ref.ldscore[k]) < 1e-2, "LD score should be the sum of adjusted r^2"
    save(fig, out, "ld-neighbourhood.svg")


def fig_joint_vs_marginal(ref, out) -> None:
    k, tagged, r2, untagged = neighbourhood(ref, INDEX_SNP)
    pos = ref.pos.astype(float) / 1e6
    half = 0.45
    beta = 0.02
    fig, axes = new_figure(2, 1, figsize=(6.8, 3.9), sharex=True)
    ax = axes[0]
    everything = np.concatenate([tagged, untagged])
    ax.plot(pos[everything], np.zeros(everything.size), "|", ms=5, color=GREY, alpha=0.6)
    ax.vlines([pos[k]], 0, [beta], color=PALETTE[2], lw=2.2)
    ax.plot([pos[k]], [beta], "D", ms=6, color=PALETTE[2])
    ax.set_ylabel("Joint effect $\\beta^{(J)}_m$")
    ax.set_ylim(-0.002, beta * 1.25)
    ax.text(0.015, 0.92, "one susceptibility SNP; all other SNPs have no effect of their own",
            transform=ax.transAxes, va="top", fontsize=8.5)
    panel_label(ax, "a")
    ax = axes[1]
    marg = beta * np.sqrt(r2)
    ax.plot(pos[untagged], np.zeros(untagged.size), "|", ms=5, color=GREY, alpha=0.6)
    ax.vlines(pos[tagged], 0, marg, color=PALETTE[1], lw=0.9, alpha=0.85)
    ax.plot(pos[tagged], marg, "o", ms=3, color=PALETTE[1])
    ax.plot([pos[k]], [beta], "D", ms=6, color=PALETTE[2])
    ax.set_ylabel("Marginal effect $|\\beta_k| = |\\rho_{km}|\\,\\beta^{(J)}_m$")
    ax.set_ylim(-0.002, beta * 1.25)
    ax.set_xlim(pos[k] - half, pos[k] + half)
    ax.set_xlabel(f"Position on chromosome {int(ref.chrom[k])} (Mb)")
    ax.text(0.015, 0.92, f"a GWAS sees an association at every one of the {tagged.size} markers in LD with it",
            transform=ax.transAxes, va="top", fontsize=8.5)
    panel_label(ax, "b")
    save(fig, out, "joint-vs-marginal.svg")


def fig_effect_prior(m2, m3, out) -> None:
    x = np.linspace(-0.035, 0.035, 1401)
    fig, axes = new_figure(1, 2, figsize=(7.2, 3.1), sharey=True)
    for ax, fit, letter in ((axes[0], m2, "a"), (axes[1], m3, "b")):
        p, sig = fit.mixture
        total = np.zeros_like(x)
        for h, (w, s2) in enumerate(zip(p, sig)):
            comp = w * norm.pdf(x, scale=np.sqrt(s2))
            total += comp
            if len(p) > 1:
                ax.plot(x, comp, color=PALETTE[1 + h], lw=1.1, ls="--",
                        label=f"cluster {h + 1}: $p_{h + 1}$ = {w:.2f}, $\\sigma_{h + 1}$ = {np.sqrt(s2):.4f}")
        ax.fill_between(x, total, color=LIGHT, alpha=0.7)
        label = f"$N(0,\\sigma^2)$, $\\sigma$ = {np.sqrt(sig[0]):.4f}" if len(p) == 1 else "mixture of both"
        ax.plot(x, total, color="black", lw=1.5, label=label)
        top = total.max()
        ax.annotate("", xy=(0, top * 1.12), xytext=(0, 0), arrowprops={"arrowstyle": "-|>", "color": PALETTE[2], "lw": 2})
        ax.text(0.0125, top * 0.98, f"point mass at 0:\n$1-\\pi_c$ = {1 - fit.pic:.4f}", color=PALETTE[2], fontsize=8.5, va="top")
        ax.set_title(f"{fit.model}: {fit.n_ssnp:,.0f} susceptibility SNPs", loc="center")
        ax.set_xlabel("Joint effect size $\\beta^{(J)}$ (per SD of genotype)")
        ax.set_ylim(0, top * 1.2)
        ax.legend(loc="upper left", fontsize=7.5, handlelength=1.6)
        panel_label(ax, letter)
    axes[0].set_ylabel("Density of non-null effects")
    save(fig, out, "effect-prior.svg")


def fig_marginal_mixture(m2, out) -> None:
    pic, sig, a = m2.estimates
    n = 130_000.0
    markers = [("small neighbourhood", 5, 2.5), ("large neighbourhood", 150, 55.0)]
    z = np.linspace(-9, 9, 1801)
    fig, axes = new_figure(2, 2, figsize=(7.0, 5.0), height_ratios=[1, 1.5])
    handles = None
    for col, (name, nstar, ld) in enumerate(markers):
        j = np.arange(0, 11)
        prior = binom.pmf(j, nstar, pic)
        ax = axes[0, col]
        ax.bar(j, prior, color=[GREY] + [PALETTE[1]] * 10, width=0.75)
        ax.set_yscale("log")
        ax.set_ylim(1e-7, 2)
        ax.set_xticks(j)
        ax.set_xlabel("Number of tagged susceptibility SNPs, $j$")
        ax.set_title(f"{name}: $N^*_k$ = {nstar}, $\\ell_k$ = {ld:g}")
        if col == 0:
            ax.set_ylabel("Prior probability")
        panel_label(ax, "ab"[col])

        ax = axes[1, col]
        total = np.zeros_like(z)
        for jj in range(0, 11):
            var_z = 1.0 + n * (jj * sig * ld / nstar + a)  # variance of z = betahat * sqrt(n)
            comp = prior[jj] * norm.pdf(z, scale=np.sqrt(var_z))
            total += comp
            if jj <= 3:
                ax.plot(z, comp, lw=1.0, color=GREY if jj == 0 else PALETTE[1], alpha=1.0 if jj == 0 else 1.1 - 0.25 * jj,
                        label={0: "$j = 0$: no susceptibility SNP tagged", 1: "$j = 1, 2, 3$ (fainter with $j$)"}.get(jj))
        ax.plot(z, total, color="black", lw=1.6, label="mixture = model for this marker")
        ax.plot(z, norm.pdf(z), color=PALETTE[2], lw=1.1, ls=":", label="$N(0,1)$: no signal, no inflation")
        ax.set_yscale("log")
        ax.set_ylim(1e-9, 1)
        ax.set_xlabel("$z$ statistic")
        if col == 0:
            ax.set_ylabel("Density")
            handles = ax.get_legend_handles_labels()
        panel_label(ax, "cd"[col])
    assert handles is not None
    fig.legend(*handles, loc="outside lower center", ncols=2, fontsize=8)
    save(fig, out, "marginal-mixture.svg")


def main() -> None:
    args = base_parser(__doc__).parse_args()
    ref = load_reference(args.reference)
    fig_ld_neighbourhood(ref, args.out)
    fig_joint_vs_marginal(ref, args.out)
    m2, m3 = load_fit(args.runs, "height.M2"), load_fit(args.runs, "height.M3")
    fig_effect_prior(m2, m3, args.out)
    fig_marginal_mixture(m2, args.out)


if __name__ == "__main__":
    main()
