"""Figures for the downstream-analysis chapters.

* ``projection.svg``   – expected discoveries and heritability explained vs sample size
* ``prs-height.svg``   – expected predictive R^2 of a polygenic score for height
* ``posterior.png``    – local false-discovery rate and shrinkage of individual markers
* ``auc-disease.svg``  – expected AUC for the simulated disease (needs
  ``<runs>/sim_disease.best.json``)
"""

from __future__ import annotations

import warnings

import numpy as np
from _common import GREY, PALETTE, base_parser, load_fit, load_reference, new_figure, panel_label, save
from scipy.stats import norm

import genesis_gwas as gg
from genesis_gwas.plotting import plot_projection


def fig_projection(m2, m3, out) -> None:
    grid = np.linspace(2e4, 2e6, 200)
    fits = {"M3 (selected)": m3, "M2": m2}
    fig, axes = new_figure(1, 2, figsize=(7.0, 3.2))
    plot_projection(fits, axes[0], what="discoveries", n_grid=grid, ci=True)
    plot_projection(fits, axes[1], what="gv", n_grid=grid, ci=True)
    for ax, letter in zip(axes, "ab"):
        ax.axvline(m3.mean_n, color=GREY, lw=0.8, ls=":")
        panel_label(ax, letter)
    axes[0].set_ylabel("Expected discoveries ($P < 5\\times10^{-8}$)")
    axes[0].text(m3.mean_n * 1.2, axes[0].get_ylim()[1] * 0.97, "sample size of\nthis GWAS", color=GREY, fontsize=8, va="top")
    axes[1].set_ylabel("Heritability explained (%)")
    save(fig, out, "projection.svg")


def fig_prs_height(m3, out, M_independent: float = 200_000) -> None:
    grid = np.logspace(np.log10(2e4), 6.5, 40)
    opt = gg.polyrisk_predict(grid, m3, M=M_independent, threshold="optimum")
    gws = gg.polyrisk_predict(grid, m3, M=M_independent, threshold="gwas")
    fig, axes = new_figure(1, 2, figsize=(7.0, 3.1))
    ax = axes[0]
    ax.plot(grid, opt.mu**2, color=PALETTE[2], lw=1.5, label="optimal threshold")
    ax.plot(grid, gws.mu**2, color=PALETTE[0], lw=1.5, label="$P < 5\\times10^{-8}$")
    ax.axhline(m3.heritability, color=GREY, lw=0.8, ls="--")
    ax.text(grid[0] * 1.1, m3.heritability * 1.01, f"$h^2$ = {m3.heritability:.3f}", color=GREY, fontsize=8, va="bottom")
    ax.set_xscale("log")
    ax.set_ylim(0, m3.heritability * 1.12)
    ax.set_xlabel("GWAS sample size")
    ax.set_ylabel("Expected predictive $R^2$ ($=\\mu^2$)")
    ax.legend(loc="lower right", fontsize=8)
    panel_label(ax, "a")
    ax = axes[1]
    ax.plot(grid, opt.alpha, color=PALETTE[1], lw=1.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("GWAS sample size")
    ax.set_ylabel("Optimal significance threshold $\\alpha$")
    panel_label(ax, "b")
    save(fig, out, "prs-height.svg")


def fig_posterior(ref, m3, sumstats, out) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = gg.prepare(sumstats, ref)
    post = gg.posterior_marginal(data, m3)
    z = post["z"].to_numpy()
    lfdr = post["p_null"].to_numpy()
    shrunk_z = post["post_mean"].to_numpy() * np.sqrt(data.n)   # posterior mean on the z scale
    rng = np.random.default_rng(1)
    order = rng.permutation(z.size)  # all markers, drawn in random order (rasterised)
    ld = np.log10(data.ldscore[order])
    fig, axes = new_figure(1, 2, figsize=(7.2, 3.4))
    ax = axes[0]
    sc = ax.scatter(np.abs(z[order]), lfdr[order], c=ld, s=1.2, cmap="viridis", linewidths=0, rasterized=True)
    ax.set_xlabel("$|z|$")
    ax.set_ylabel("Local false-discovery rate")
    ax.set_xlim(0, 9.2)
    ax.set_ylim(-0.02, 1.02)
    panel_label(ax, "a")
    ax = axes[1]
    ax.plot([-9.2, 9.2], [-9.2, 9.2], color=GREY, lw=0.8, ls="--")
    ax.scatter(z[order], shrunk_z[order], c=ld, s=1.2, cmap="viridis", linewidths=0, rasterized=True)
    ax.set_xlabel("Observed $z$")
    ax.set_ylabel("Posterior mean ($z$ scale)")
    ax.set_xlim(-9.2, 9.2)
    ax.set_ylim(-9.2, 9.2)
    panel_label(ax, "b")
    cb = fig.colorbar(sc, ax=axes, shrink=0.85, pad=0.02)
    cb.set_label("LD score $\\ell_k$")
    ticks = [1, 3, 10, 30, 100]
    cb.set_ticks(np.log10(ticks), labels=[str(t) for t in ticks])
    for thr in (0.5, 0.05, 0.01):
        print(f"markers with lfdr < {thr}: {int((lfdr < thr).sum()):,}")
    print(f"markers with P < 5e-8: {int((np.abs(z) > 5.4513).sum()):,};  mean lfdr: {lfdr.mean():.4f}")
    save(fig, out, "posterior.png")


def fig_auc_disease(runs, out, M_independent: float = 200_000) -> None:
    path = runs / "sim_disease.best.json"
    if not path.exists():
        print(f"skipping auc-disease.svg: {path} not found")
        return
    fit = gg.FitResult.from_json(path)
    grid = np.logspace(4, 6.5, 40)
    opt = gg.polyrisk_predict(grid, fit, M=M_independent, threshold="optimum", k_fold=[2])
    gws = gg.polyrisk_predict(grid, fit, M=M_independent, threshold="gwas", k_fold=[2])
    ceiling = float(norm.cdf(np.sqrt(fit.heritability / 2.0)))
    fig, axes = new_figure(1, 2, figsize=(7.0, 3.1))
    ax = axes[0]
    ax.plot(grid, opt.auc, color=PALETTE[2], lw=1.5, label="optimal threshold")
    ax.plot(grid, gws.auc, color=PALETTE[0], lw=1.5, label="$P < 5\\times10^{-8}$")
    ax.axhline(ceiling, color=GREY, lw=0.8, ls="--")
    ax.text(grid[0] * 1.1, ceiling + 0.002, f"ceiling {ceiling:.3f}", color=GREY, fontsize=8, va="bottom")
    ax.set_xscale("log")
    ax.set_ylim(0.5, ceiling + 0.02)
    ax.set_xlabel("Effective GWAS sample size")
    ax.set_ylabel("Expected AUC")
    ax.legend(loc="lower right", fontsize=8)
    panel_label(ax, "a")
    ax = axes[1]
    ax.plot(grid, 100 * np.asarray(opt.pci)[:, 0], color=PALETTE[2], lw=1.5, label="future cases identified")
    ax.plot(grid, 100 * np.asarray(opt.ppi)[:, 0], color=PALETTE[1], lw=1.5, label="population selected")
    ax.set_xscale("log")
    ax.set_ylim(0, None)
    ax.set_xlabel("Effective GWAS sample size")
    ax.set_ylabel("At twice the average risk or more (%)")
    ax.legend(loc="upper left", fontsize=8)
    panel_label(ax, "b")
    print(f"disease AUC ceiling {ceiling:.4f}; AUC at 5.2e4 = {float(gg.polyrisk_predict(52_000, fit, M=M_independent).auc):.3f}")
    save(fig, out, "auc-disease.svg")


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--sumstats", default=None, help="height summary statistics [the reference's example file]")
    parser.add_argument("--skip-posterior", action="store_true", help="skip the per-SNP posterior figure")
    args = parser.parse_args()
    m2, m3 = load_fit(args.runs, "height.M2"), load_fit(args.runs, "height.M3")
    fig_projection(m2, m3, args.out)
    fig_prs_height(m3, args.out)
    fig_auc_disease(args.runs, args.out)
    if not args.skip_posterior:
        ref = load_reference(args.reference)
        fig_posterior(ref, m3, args.sumstats or ref.example_path(), args.out)


if __name__ == "__main__":
    main()
