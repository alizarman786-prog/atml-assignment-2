from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = Path("results")
OUT = Path("report/figures")
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
BLUE, GREEN, RED, AMBER, GREY = "#2f5aa8", "#1f7a5a", "#b3321f", "#946200", "#5d6672"


def jload(path):
    return json.load(open(path))


def jlines(path):
    return pd.read_json(path, lines=True)


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def run_panels(panels):
    """panels: list of (ax, title, fn). A failing panel prints its error and leaves a note instead of stopping."""
    for ax, title, fn in panels:
        try:
            fn(ax)
            ax.set_title(title, loc="left")
        except Exception as e:
            print("PANEL FAILED:", title, "->", repr(e))
            ax.text(0.5, 0.5, "panel failed\n" + repr(e)[:60], ha="center", va="center", transform=ax.transAxes)


def finish(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / (name + ".pdf"))
    fig.savefig(OUT / (name + ".png"), dpi=200)
    plt.close(fig)
    print("saved", OUT / (name + ".pdf"))
