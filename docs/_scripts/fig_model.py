"""Figures for the theory chapters.

* ``truncation-mass.svg``         – prior mass discarded by truncating the latent counts at c0
* ``truncation-sensitivity.svg``  – estimates against the truncation level (needs
  ``<runs>/truncation.json`` written by ``examples/truncation.py``)
* ``sandwich-neighbourhood.svg``  – which pairs of markers enter the sandwich "meat"
"""

from __future__ import annotations

import json

import numpy as np
from _common import GREY, PALETTE, base_parser, load_fit, load_reference, new_figure, panel_label, save
from scipy.stats import binom


def fig_truncation(ref, m2, m3, out) -> None:
    fig, axes = new_figure(1, 2, figsize=(7.0, 3.1))
    # (a) tail mass against the expected number of tagged susceptibility SNPs
    ax = axes[0]
    mean = np.logspace(-1.5, 1.5, 300)
    pic = 0.01
    for c0, color in ((10, PALETTE[1]), (20, PALETTE[2])):
        nstar = np.maximum(np.round(mean / pic), c0 + 1)
        tail = binom.sf(c0, nstar, mean / nstar)
        ax.plot(mean, tail, color=color, lw=1.5, label=f"$c_0$ = {c0}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-16, 2)
    ax.set_xlabel("Expected number of tagged susceptibility SNPs, $N^*_k\\,\\pi_c$")
    ax.set_ylabel("Discarded prior mass, $1-y_k$")
    ax.axhline(1e-6, color=GREY, lw=0.7, ls=":")
    ax.text(0.035, 2.2e-6, "$10^{-6}$", color=GREY, fontsize=8)
    ax.legend(loc="lower right")
    panel_label(ax, "a")

    # (b) where real markers sit: distribution of N* pi_c in the reference for two fits
    ax = axes[1]
    nstar = ref.nstar.astype(float)
    bins = np.logspace(-2.5, 1.0, 60)
    for fit, color in ((m2, PALETTE[1]), (m3, PALETTE[2])):
        ax.hist(nstar * fit.pic, bins=bins, histtype="step", color=color, lw=1.3,
                label=f"height {fit.model}: $\\pi_c$ = {fit.pic:.4f}")
        trunc = fit.diagnostics["truncation"]  # computed by the package for the fitted markers
        print(f"{fit.model}: c0 = {trunc['c0']}, mean discarded mass {trunc['mean_tail']:.3e}, "
              f"max {trunc['max_tail']:.3e}, fraction of markers above 1e-6: {trunc['frac_above_1e6']:.4f}, "
              f"log-likelihood gap {trunc['log_mass_deficit']:.4f}")
    ax.set_xscale("log")
    ax.set_xlabel("$N^*_k\\,\\pi_c$ for the reference SNPs")
    ax.set_ylabel("Number of SNPs")
    ax.legend(loc="upper left")
    panel_label(ax, "b")
    save(fig, out, "truncation-mass.svg")


def fig_truncation_sensitivity(runs, m2, m3, out) -> None:
    path = runs / "truncation.json"
    if not path.exists():
        print(f"skipping truncation-sensitivity.svg: {path} not found (run examples/truncation.py)")
        return
    data = json.loads(path.read_text())
    fig, axes = new_figure(1, 2, figsize=(7.2, 3.1))

    ax = axes[0]
    for key, fit, color in (("height_M2", m2, PALETTE[1]), ("height_M3", m3, PALETTE[2])):
        rows = data[key]
        c0 = np.array([r["c0"] for r in rows])
        pic = np.array([r["pic"] for r in rows])
        rel = 100 * (pic / pic[-1] - 1)
        ax.plot(c0, rel, "-o", ms=4, color=color, lw=1.4, label=f"{fit.model}")
        chosen = c0 == fit.c0
        ax.plot(c0[chosen], rel[chosen], "o", ms=9, mfc="none", mec=color, mew=1.4)
        print(f"{key}: default c0 = {fit.c0}; pic relative to c0 = {c0[-1]}: "
              + ", ".join(f"{c}: {v:+.3f}%" for c, v in zip(c0, rel)))
    ax.axhline(0, color=GREY, lw=0.7, ls=":")
    ax.set_xticks(c0)
    ax.set_xlabel("Truncation level $c_0$")
    ax.set_ylabel("$\\hat\\pi_c$ relative to $c_0$ = 20 (%)")
    ax.set_title("Full height GWAS")
    ax.legend(loc="lower right")
    panel_label(ax, "a")

    ax = axes[1]
    rows = data["chr22_M2"]
    c0 = np.array([r["c0"] for r in rows])
    pic = np.array([r["pic"] for r in rows])
    se = np.array([r["pic_se"] for r in rows])
    ax.errorbar(c0, pic, yerr=se, fmt="-o", ms=4, color=PALETTE[1], lw=1.4, capsize=2.5, label="$\\hat\\pi_c$ ± SE")
    ax.set_ylim(0, None)
    ax.set_xlabel("Truncation level $c_0$")
    ax.set_ylabel("$\\hat\\pi_c$")
    ax.set_title("Chromosome 22 only")
    twin = ax.twinx()
    twin.plot(c0, [r["h2"] for r in rows], "-s", ms=4, color=PALETTE[2], lw=1.4, label="heritability")
    twin.set_ylim(0, 0.012)
    twin.set_ylabel("Heritability (chromosome 22)")
    twin.spines["top"].set_visible(False)
    handles = ax.get_legend_handles_labels()[0] + twin.get_legend_handles_labels()[0]
    ax.legend(handles, [h.get_label() for h in handles], loc="upper left")
    panel_label(ax, "b")
    save(fig, out, "truncation-sensitivity.svg")


def fig_sandwich(ref, out) -> None:
    """Sparsity pattern of the tagging relation for a stretch of consecutive markers."""
    nb = ref.neighborhoods
    start = int(ref.lookup(["rs724016"])[0]) - 150
    size = 300
    rows, cols = [], []
    for i in range(start, start + size):
        idx = np.asarray(nb.indices[nb.indptr[i] : nb.indptr[i + 1]])
        idx = idx[(idx >= start) & (idx < start + size)]
        rows.append(np.full(idx.size, i - start))
        cols.append(idx - start)
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    k = 150
    mine = cols[rows == k]
    image = np.zeros((size, size), dtype=np.uint8)
    image[rows, cols] = 1
    image[k, mine] = 2
    fig, axes = new_figure(1, 2, figsize=(7.0, 3.4), width_ratios=[1.0, 1.25])
    ax = axes[0]
    from matplotlib.colors import ListedColormap

    ax.imshow(image, cmap=ListedColormap(["white", PALETTE[1], PALETTE[2]]), interpolation="nearest", vmin=0, vmax=2)
    ax.axhline(k, color=PALETTE[2], lw=0.6, alpha=0.6)
    ax.set_xlabel("marker $k'$ (consecutive along the chromosome)")
    ax.set_ylabel("marker $k$")
    ax.set_title("pairs $(k, k')$ with $k' \\in \\mathcal{N}_k$", fontsize=9.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(True)
    panel_label(ax, "a")

    ax = axes[1]
    pos = (ref.pos[start : start + size].astype(float)) / 1e6
    ax.plot(pos, np.zeros(size), "|", color=GREY, ms=7)
    ax.plot(pos[mine], np.zeros(mine.size), "|", color=PALETTE[2], ms=12, markeredgewidth=1.4)
    ax.plot([pos[k]], [0], "D", color=PALETTE[2], ms=6)
    ax.annotate("marker $k$", xy=(pos[k], 0.0), xytext=(pos[k], 0.5), ha="center", fontsize=9,
                arrowprops={"arrowstyle": "-", "color": PALETTE[2], "lw": 0.8})
    lo, hi = pos[mine].min(), pos[mine].max()
    ax.annotate("", xy=(lo, -0.32), xytext=(hi, -0.32), arrowprops={"arrowstyle": "|-|", "color": PALETTE[2], "lw": 1.2})
    ax.text((lo + hi) / 2, -0.4, f"$\\bar U_k = \\sum_{{k' \\in \\mathcal{{N}}_k}} U_{{k'}}$   ({mine.size} markers)",
            ha="center", va="top", fontsize=9.5, color=PALETTE[2])
    ax.set_ylim(-1.0, 1.0)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(f"Position on chromosome {int(ref.chrom[start])} (Mb)")
    ax.set_title("the neighbourhood of one marker", fontsize=9.5)
    panel_label(ax, "b")
    save(fig, out, "sandwich-neighbourhood.svg")


def main() -> None:
    args = base_parser(__doc__).parse_args()
    ref = load_reference(args.reference)
    m2, m3 = load_fit(args.runs, "height.M2"), load_fit(args.runs, "height.M3")
    fig_truncation(ref, m2, m3, args.out)
    fig_truncation_sensitivity(args.runs, m2, m3, args.out)
    fig_sandwich(ref, args.out)
    for model in ("M2", "M3"):                       # numbers quoted in the text: level held at c0 = 10
        path = args.runs / f"height_em_newton.{model}.json"
        if path.exists():
            import genesis_gwas as gg

            trunc = gg.FitResult.from_json(path).diagnostics["truncation"]
            print(f"{model} at fixed c0 = {trunc['c0']}: mean discarded mass {trunc['mean_tail']:.3e}, max "
                  f"{trunc['max_tail']:.3e}, fraction above 1e-6: {trunc['frac_above_1e6']:.4f}, "
                  f"log-likelihood gap {trunc['log_mass_deficit']:.4f}")


if __name__ == "__main__":
    main()
