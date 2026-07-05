"""
figstyle.py — shared editorial figure style.

One module every figure imports, so the whole set reads as one hand.
Modern-editorial direction: off-white canvas, ink + one warm accent,
large restrained type, generous whitespace, no emoji/callouts/traffic-light color.
"""

import matplotlib.pyplot as plt
from matplotlib import rcParams

# ── Palette ───────────────────────────────────────────────────────────────────
INK      = "#1A1A1A"   # near-black for text and primary marks
PAPER    = "#FBFAF7"   # warm off-white canvas
MUTED    = "#9A9791"   # secondary / de-emphasised
GRID     = "#E7E4DD"   # hairline grid
ACCENT   = "#C2703D"   # the single warm accent (terracotta)
ACCENT_2 = "#3C5A78"   # cool secondary, used sparingly for the second class

# Two-class encoding: keep ONE warm + ONE cool, never red-vs-blue-vs-green
DCM = INK       # DCM = the reference, drawn in ink
ACM = ACCENT    # ACM = the thing of interest, drawn in accent


def apply():
    """Set global rcParams. Call once at the top of a figure script."""
    rcParams.update({
        "font.family": "Helvetica",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "regular",
        "axes.labelsize": 10.5,
        "axes.labelcolor": INK,
        "text.color": INK,
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "axes.grid": False,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "figure.dpi": 160,
    })


def clean(ax, keep=("left", "bottom")):
    """Strip chartjunk: drop unwanted spines, thin the rest."""
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)
    for side in keep:
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(length=3, width=0.8)
    return ax


def title(ax, headline, sub=None):
    """Left-aligned editorial title: bold headline, muted subline above the axes."""
    ax.set_title("")  # clear default
    ax.text(0.0, 1.10, headline, transform=ax.transAxes,
            fontsize=13, fontweight="bold", ha="left", va="bottom", color=INK)
    if sub:
        ax.text(0.0, 1.045, sub, transform=ax.transAxes,
                fontsize=9.5, ha="left", va="bottom", color=MUTED)


def figure_caption(fig, text, y=0.005):
    """Caption carries the explanation — text lives here, not inside the plot."""
    fig.text(0.5, y, text, ha="center", va="bottom",
             fontsize=9, color=MUTED, wrap=True)
