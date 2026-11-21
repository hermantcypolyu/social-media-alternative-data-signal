"""Shared notebook helpers: consistent plotting style and output paths.

Keeping this in ``src`` rather than duplicated across notebooks means a single
place defines figure resolution, palette and the output folders. It contains no
research logic.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "outputs" / "figures"
TABLES = ROOT / "outputs" / "tables"
PROCESSED = ROOT / "data" / "processed"

PALETTE = {
    "primary": "#1f3a5f",
    "accent": "#c8553d",
    "muted": "#8d99ae",
    "positive": "#2a9d8f",
    "negative": "#c8553d",
    "neutral": "#8d99ae",
}


def setup_style() -> None:
    """Apply a consistent, print-friendly matplotlib style."""
    matplotlib.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 11.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "-",
            "legend.frameon": False,
            "figure.facecolor": "white",
        }
    )


def ensure_output_dirs() -> None:
    """Create the output directories if a fresh clone does not have them."""
    for d in (FIGURES, TABLES, PROCESSED):
        d.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, name: str, note: str | None = None) -> None:
    """Save a figure to ``outputs/figures`` and print its repo-relative path.

    Returns ``None`` deliberately. Returning the absolute ``Path`` made Jupyter
    echo ``PosixPath('/home/<user>/...')`` as a cell result, which leaked a local
    filesystem path into the committed notebook.

    Parameters
    ----------
    fig
        Figure to save.
    name
        Filename without extension.
    note
        Optional provenance note appended as a footer. Used to label synthetic
        or engineering-validation figures unambiguously.
    """
    ensure_output_dirs()
    if note:
        fig.text(0.005, 0.005, note, fontsize=7, color="#555555", ha="left", va="bottom")
    path = FIGURES / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"saved {path.relative_to(ROOT)}")


def save_table(df, name: str) -> None:
    """Save a DataFrame to ``outputs/tables`` as CSV.

    Returns ``None`` for the same reason as :func:`save_figure`.
    """
    ensure_output_dirs()
    path = TABLES / f"{name}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"saved {path.relative_to(ROOT)}")