"""
actuator/plotstyle.py -- one look for every SysID figure, with a validated palette.

Colours are the data-viz reference palette's categorical slots 1-3, checked with its
validator for all-pairs colour-vision separation on a light surface (worst CVD dE 9.2,
worst normal-vision dE 24.0, both above the 8/15 floors). Do not substitute eyeballed
hexes: an overlay chart puts every pair in play at once, and only the first three slots
clear that bar.

Roles, not entities: TRUTH is always the hardware recording, SIM1/SIM2 the models being
tested against it. So blue is "what really happened" in every figure in this directory.

Slot 3 (aqua) sits just under 3:1 against the surface, which the validator flags as
needing relief -- every panel carries a visible MAE label and a legend, which is it.
"""
from __future__ import annotations

SURFACE = "#fcfcfb"
INK = "#0b0b0b"          # primary text
INK2 = "#52514e"         # secondary text
MUTED = "#8a8884"        # goal / reference line
GRID = "#e8e7e3"         # hairline grid, one shade off the surface

TRUTH = "#2a78d6"        # slot 1, blue   -- recorded on hardware
SIM1 = "#eb6834"         # slot 2, orange -- first model under test
SIM2 = "#1baf7a"         # slot 3, aqua   -- second model under test

LW = 2.0                 # data lines
LW_REF = 1.2             # goal / reference


def apply() -> None:
    """Set rcParams. Call once per process, before creating figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.facecolor": SURFACE, "axes.edgecolor": GRID, "axes.linewidth": 0.8,
        "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "grid.linestyle": "-",            # solid hairline: dashes read as a threshold
        "xtick.color": INK2, "ytick.color": INK2,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.titlesize": 9, "axes.labelsize": 8,
        "legend.frameon": False, "legend.fontsize": 9,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "figure.dpi": 110, "savefig.dpi": 140, "savefig.bbox": "tight",
    })


def panel_note(ax, text: str, loc: str = "upper left") -> None:
    """A small muted annotation inside a panel (the MAE label, usually).

    Text wears ink, never a series colour -- the coloured line beside it carries identity.
    """
    xy = {"upper left": (0.02, 0.97), "upper right": (0.98, 0.97),
          "lower left": (0.02, 0.03), "lower right": (0.98, 0.03)}[loc]
    ax.text(*xy, text, transform=ax.transAxes, fontsize=7.5, color=INK2,
            ha="right" if "right" in loc else "left",
            va="top" if "upper" in loc else "bottom")


def shade_torque_off(ax, t, torque_enable) -> None:
    """Mark the spans where the servo was unpowered (lift_and_drop's free fall).

    Those samples are not tracking error -- nothing is driving the joint -- so a reader
    comparing curves there needs to know. Drawn behind the data.
    """
    import numpy as np
    off = ~np.asarray(torque_enable, dtype=bool)
    if not off.any():
        return
    edges = np.diff(off.astype(int))
    starts = list(np.flatnonzero(edges == 1) + 1) + ([0] if off[0] else [])
    ends = list(np.flatnonzero(edges == -1) + 1) + ([len(off)] if off[-1] else [])
    for s, e in zip(sorted(starts), sorted(ends)):
        ax.axvspan(t[s], t[min(e, len(t) - 1)], color=GRID, alpha=0.55, lw=0, zorder=0)
