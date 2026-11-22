"""Shared helpers for the documentation figures.

Every figure script is a small command-line program::

    python docs/_scripts/fig_concepts.py --out docs/_static/figures [--reference NAME_OR_PATH]

Nothing here hard-codes a location: the LD reference is resolved by
``genesis_gwas.load_reference`` (so ``$GENESIS_DATA_DIR`` applies), and the fitted
models are read from ``--runs`` (the directory written by ``run_examples.sh``).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from genesis_gwas.plotting import PALETTE, style_axes  # noqa: E402

__all__ = [
    "GREY",
    "PALETTE",
    "base_parser",
    "load_fit",
    "load_reference",
    "new_figure",
    "save",
]

GREY = "#8C8C8C"
LIGHT = "#D9D9D9"

matplotlib.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9.5,
        "legend.fontsize": 8.5,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "svg.hashsalt": "genesis-docs",  # reproducible SVG ids
        "mathtext.fontset": "dejavusans",
    }
)


def base_parser(description: str) -> argparse.ArgumentParser:
    """Arguments common to all figure scripts."""
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "--out",
        type=Path,
        default=here.parent / "_static" / "figures",
        help="directory for the figures [docs/_static/figures]",
    )
    p.add_argument(
        "--reference",
        default=os.environ.get("GENESIS_DOCS_REFERENCE", "eur_1kg_hm3"),
        help="LD reference name or bundle directory [$GENESIS_DOCS_REFERENCE or eur_1kg_hm3]",
    )
    p.add_argument(
        "--runs",
        type=Path,
        default=Path(os.environ.get("GENESIS_DOCS_RUNS", "results")),
        help="directory with the fitted models written by run_examples.sh "
        "[$GENESIS_DOCS_RUNS or ./results]",
    )
    return p


def load_reference(name_or_path: str, **kwargs: Any) -> Any:
    """Load the LD reference used for the figures."""
    import genesis_gwas as gg

    return gg.load_reference(name_or_path, **kwargs)


def load_fit(runs: Path, name: str) -> Any:
    """Read ``<runs>/<name>.json`` as a :class:`genesis_gwas.FitResult`."""
    import genesis_gwas as gg

    path = Path(runs) / f"{name}.json"
    if not path.exists():
        raise SystemExit(f"{path} not found: run docs/_scripts/run_examples.sh first (see README).")
    return gg.FitResult.from_json(path)


def new_figure(
    nrows: int = 1, ncols: int = 1, figsize: tuple[float, float] = (6.6, 3.2), **kwargs: Any
) -> tuple[Figure, Any]:
    """A constrained-layout figure with styled axes (no pyplot state)."""
    fig = Figure(figsize=figsize, layout="constrained")
    axes = fig.subplots(nrows, ncols, **kwargs)
    for ax in fig.axes:
        style_axes(ax)
    return fig, axes


def panel_label(ax: Axes, letter: str) -> None:
    """Bold panel letter in the top-left corner."""
    ax.text(-0.02, 1.06, letter, transform=ax.transAxes, fontweight="bold", fontsize=11, va="bottom", ha="right")


def save(fig: Figure, out: Path, name: str, dpi: int = 200) -> Path:
    """Write ``fig`` to ``out/name`` (format from the extension) and report its size."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    kwargs: dict[str, Any] = {"dpi": dpi}
    if path.suffix == ".svg":
        kwargs["metadata"] = {"Date": None}  # reproducible output
    fig.savefig(path, **kwargs)
    print(f"wrote {path}  ({path.stat().st_size / 1024:.0f} kB)")
    preview = os.environ.get("GENESIS_DOCS_PREVIEW")  # optional PNG copies for quick inspection
    if preview and path.suffix == ".svg":
        Path(preview).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(preview) / (path.stem + ".png"), dpi=110)
    return path
