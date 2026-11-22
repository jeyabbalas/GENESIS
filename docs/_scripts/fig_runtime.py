"""``runtime.svg`` – wall-clock comparison with the original software on one machine.

The numbers are measurements recorded in ``runtime_measurements.json`` next to this
script, together with the machine and the conditions. Edit that file when you re-measure;
nothing is timed here.
"""

from __future__ import annotations

import json
from pathlib import Path

from _common import GREY, PALETTE, base_parser, new_figure, save

COLOURS = {"original": GREY, "legacy": PALETTE[1], "default": PALETTE[2]}


def fmt(seconds: float) -> str:
    if seconds >= 120:
        return f"{seconds / 60:.0f} min" if seconds >= 600 else f"{seconds / 60:.1f} min"
    return f"{seconds:.0f} s" if seconds >= 20 else f"{seconds:.1f} s"


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--measurements", type=Path, default=Path(__file__).with_name("runtime_measurements.json"))
    args = parser.parse_args()
    spec = json.loads(args.measurements.read_text())
    groups = spec["groups"]

    fig, ax = new_figure(figsize=(7.0, 0.36 * sum(len(g["bars"]) + 1.3 for g in groups) + 0.7))
    y = 0.0
    ticks, labels = [], []
    for group in groups:
        ax.text(1.02, y - 0.75, group["title"], fontsize=9.5, fontweight="bold", va="center", ha="left")
        reference = group["bars"][0]["seconds"]
        for bar in group["bars"]:
            ax.barh(y, bar["seconds"], height=0.62, color=COLOURS[bar["kind"]])
            note = fmt(bar["seconds"])
            if bar["kind"] != "original" and len(group["bars"]) > 1:
                note += f"   ({reference / bar['seconds']:.0f}× faster)"
            ax.text(bar["seconds"] * 1.08, y, note, va="center", fontsize=8.5)
            ticks.append(y)
            labels.append(bar["label"])
            y += 1.0
        y += 1.3
    ax.set_yticks(ticks, labels)
    ax.set_ylim(y - 1.6, -1.3)
    ax.set_xscale("log")
    ax.set_xlim(1.0, max(b["seconds"] for g in groups for b in g["bars"]) * 12)
    ax.set_xlabel("Wall-clock time (seconds, logarithmic scale)")
    ax.tick_params(axis="y", length=0)
    print("\n".join(f"{g['title']}: " + ", ".join(f"{b['label']} {b['seconds']:g} s" for b in g["bars"]) for g in groups))
    save(fig, args.out, "runtime.svg")


if __name__ == "__main__":
    main()
