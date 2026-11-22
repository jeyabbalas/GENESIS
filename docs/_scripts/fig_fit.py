"""Figures for the fitting and interpretation chapters.

* ``em-trajectory.svg``   – the EM run to its own stopping rule, and where Newton's method takes over
  (needs ``<runs>/height_em.M{2,3}.json`` and ``height_em_newton.M{2,3}.json`` written by
  ``examples/em_only.py``)
* ``qq-height.png``       – model-based QQ plots, M2 vs M3
* ``effect-density.svg``  – fitted effect-size densities, M2 vs M3
* ``ld-sensitivity.svg``  – estimates under the nine LD configurations (needs
  ``<runs>/ld_sensitivity.json`` written by ``ld_sensitivity.py``)
"""

from __future__ import annotations

import json
import warnings

import numpy as np
from _common import GREY, PALETTE, base_parser, load_fit, load_reference, new_figure, panel_label, save

import genesis_gwas as gg
from genesis_gwas.plotting import plot_effect_density, qq_plot


def fig_em_trajectory(runs, out) -> None:
    """Plain EM (run to its own stopping rule) against the default EM warm-up + Newton, at c0 = 10."""
    models = ("M2", "M3")
    slow_paths = [runs / f"height_em.{m}.json" for m in models]
    fast_paths = [runs / f"height_em_newton.{m}.json" for m in models]
    if not all(path.exists() for path in slow_paths + fast_paths):
        print("skipping em-trajectory.svg: run examples/em_only.py first (section 'slow')")
        return
    fig, axes = new_figure(2, 2, figsize=(7.0, 4.7), sharex="col")
    for col, (slow_path, fast_path) in enumerate(zip(slow_paths, fast_paths)):
        em = gg.FitResult.from_json(slow_path)               # accelerate=False, c0 fixed at 10
        fit = gg.FitResult.from_json(fast_path)              # default optimiser, c0 fixed at 10
        hist = em.history
        it = np.array([h["iter"] for h in hist])
        ll = np.array([h["loglik"] for h in hist])           # at the parameters each iteration starts from
        pic = np.array([h["theta"][0] for h in hist])        # after the update of that iteration
        n_em = em.n_iter
        stop_gap = em.loglik - em.diagnostics["polish"]["em_loglik"]
        stop_pic = em.diagnostics["polish"]["em_estimates"][0]
        pol = fit.diagnostics["polish"]                      # Newton after a short warm-up
        n_warm = fit.n_iter
        warm_gap = fit.loglik - pol["em_loglik"]
        warm_pic = pol["em_estimates"][0]
        print(f"{em.model}: plain EM stops after {n_em} iterations, {stop_gap:.3f} below the maximum, "
              f"pic = {stop_pic:.6e} (maximiser {em.pic:.6e}); default: {n_warm} EM iterations "
              f"({warm_gap:.1f} below) + {pol['n_steps']} Newton steps")

        ax = axes[0, col]
        gap = em.loglik - ll
        ax.plot(it[gap > 0], gap[gap > 0], color=PALETTE[1], lw=1.4)
        ax.plot([n_em], [stop_gap], "o", ms=6, mfc="white", mec=PALETTE[1], mew=1.4)
        ax.annotate(f"EM stopping rule:\nstill {stop_gap:.2f} below", xy=(n_em, stop_gap), xytext=(0.60, 0.50),
                    textcoords="axes fraction", fontsize=8, arrowprops={"arrowstyle": "-", "color": GREY, "lw": 0.7})
        ax.plot([n_warm], [warm_gap], "D", ms=5, color=PALETTE[2])
        ax.annotate(f"default: Newton's method from here\n({pol['n_steps']} steps to the maximum)", xy=(n_warm, warm_gap),
                    xytext=(0.22, 0.86), textcoords="axes fraction", fontsize=8,
                    arrowprops={"arrowstyle": "-", "color": GREY, "lw": 0.7})
        ax.set_yscale("log")
        ax.set_title(f"{em.model}: EM alone needs {n_em} iterations")
        if col == 0:
            ax.set_ylabel("Log-likelihood below\nthe maximum")
        panel_label(ax, "ab"[col])

        ax = axes[1, col]
        ax.plot(np.concatenate([[0], it + 1]), np.concatenate([[em.starting[0]], pic]) * 1e3, color=PALETTE[1], lw=1.4,
                label="EM iterates")
        ax.axhline(em.pic * 1e3, color=GREY, lw=0.9, ls="--", label="maximiser")
        ax.plot([n_em], [stop_pic * 1e3], "o", ms=6, mfc="white", mec=PALETTE[1], mew=1.4, label="EM stopping rule")
        ax.annotate("", xy=(n_warm, fit.pic * 1e3), xytext=(n_warm, warm_pic * 1e3),
                    arrowprops={"arrowstyle": "-|>", "color": PALETTE[2], "lw": 1.8, "shrinkA": 0, "shrinkB": 0})
        ax.plot([], [], color=PALETTE[2], lw=1.8, label="Newton's method (default)")
        ax.set_xlabel("EM iteration")
        if col == 0:
            ax.set_ylabel("$\\pi_c \\times 10^{3}$")
        if col == 0:
            ax.legend(loc="upper right", fontsize=8)
        ax.set_xlim(-0.02 * n_em, n_em * 1.06)
        panel_label(ax, "cd"[col])
    save(fig, out, "em-trajectory.svg")


def fig_qq(ref, m2, m3, sumstats, out, n_sim: int = 100) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = gg.prepare(sumstats, ref)
    fig, axes = new_figure(1, 2, figsize=(7.0, 3.5))
    for ax, fit, letter in ((axes[0], m2, "a"), (axes[1], m3, "b")):
        qq = gg.qq_simulation(data, fit, ref, n_sim=n_sim, coverage=0.8, seed=123)
        qq_plot(qq, ax, axis_max=10, title=f"{fit.model}")
        print(f"{fit.model}: lambda_obs = {qq.lambda_obs:.3f}, lambda_fit = {qq.lambda_mean:.3f} "
              f"[{qq.lambda_lower:.3f}, {qq.lambda_upper:.3f}], {qq.n_sim} simulations, LD errors: {qq.ld_errors}")
        panel_label(ax, letter)
    save(fig, out, "qq-height.png")


def fig_effect_density(m2, m3, out) -> None:
    fig, axes = new_figure(1, 2, figsize=(7.2, 3.2))
    fits = {"M2 (one normal)": m2, "M3 (two normals)": m3}
    plot_effect_density(fits, axes[0], xlim=(-0.03, 0.03))
    axes[0].set_ylabel("Density among susceptibility SNPs")
    axes[0].set_xlabel("Joint effect size")
    if axes[0].get_legend() is not None:
        axes[0].get_legend().remove()          # the legend of panel b serves both
    panel_label(axes[0], "a")

    ax = axes[1]
    x = np.linspace(0.0, 0.045, 400)
    for (label, fit), color in zip(fits.items(), ("black", PALETTE[1])):
        count = np.array([float(gg.n_in_interval(v, np.inf, fit)) for v in x])
        ax.plot(x, count, color=color, lw=1.6, label=label)
    ax.set_yscale("log")
    ax.set_ylim(1, 3e4)
    ax.set_xlim(0, 0.045)
    ax.set_xlabel("Absolute joint effect size, $x$")
    ax.set_ylabel("Susceptibility SNPs with |effect| > $x$")
    ax.legend(loc="upper right")
    panel_label(ax, "b")
    save(fig, out, "effect-density.svg")


def fig_ld_sensitivity(runs, out) -> None:
    path = runs / "ld_sensitivity.json"
    if not path.exists():
        print(f"skipping ld-sensitivity.svg: {path} not found")
        return
    rows = json.loads(path.read_text())
    windows = sorted({r["window_kb"] for r in rows})
    cutoffs = sorted({r["r2_min"] for r in rows})
    fig, axes = new_figure(1, 3, figsize=(7.2, 3.2))
    panels = (("n_ssnp", "n_ssnp_se", "Susceptibility SNPs", 1.0),
              ("h2", "h2_se", "Heritability $h^2$", 1.0),
              ("mean_nstar", None, "Mean $N^*_k$ of the markers", 1.0))
    for ax, (key, se_key, label, scale), letter in zip(axes, panels, "abc"):
        for i, r2 in enumerate(cutoffs):
            sel = sorted((r for r in rows if r["r2_min"] == r2), key=lambda r: r["window_kb"])
            x = np.arange(len(windows)) + (i - 1) * 0.16
            y = np.array([r[key] for r in sel]) * scale
            err = None if se_key is None else 1.96 * np.array([r[se_key] for r in sel]) * scale
            ax.errorbar(x, y, yerr=err, fmt="o", ms=4.5, color=PALETTE[1 + i], capsize=2.5, lw=1.0,
                        label=f"$r^2 \\geq {r2:g}$")
        ax.set_xticks(np.arange(len(windows)), [f"{w / 1000:g} Mb" for w in windows])
        ax.set_xlabel("LD window")
        ax.set_ylabel(label)
        ax.set_xlim(-0.5, len(windows) - 0.5)
        panel_label(ax, letter)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=len(cutoffs), fontsize=8.5, title="tagging threshold")
    save(fig, out, "ld-sensitivity.svg")


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--sumstats", default=None, help="height summary statistics [the reference's example file]")
    parser.add_argument("--qq-sims", type=int, default=100)
    parser.add_argument("--skip-qq", action="store_true", help="skip the (slow) QQ simulation figure")
    args = parser.parse_args()
    m2, m3 = load_fit(args.runs, "height.M2"), load_fit(args.runs, "height.M3")
    fig_em_trajectory(args.runs, args.out)
    fig_effect_density(m2, m3, args.out)
    fig_ld_sensitivity(args.runs, args.out)
    if not args.skip_qq:
        ref = load_reference(args.reference)
        fig_qq(ref, m2, m3, args.sumstats or ref.example_path(), args.out, args.qq_sims)


if __name__ == "__main__":
    main()
