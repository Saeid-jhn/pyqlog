"""Shared matplotlib/seaborn styling and figure-saving helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

# Default raster resolution. Previously qlog/pcap used 900 and iperf used 300;
# 300 dpi is publication quality without the huge file sizes 900 produced.
PNG_DPI: int = 300

VALID_FORMATS = frozenset({"png", "svg", "pdf"})

_STYLE_APPLIED = False


def apply_style() -> None:
    """Apply the shared plot theme once per process (idempotent)."""
    global _STYLE_APPLIED
    if _STYLE_APPLIED:
        return
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
    })
    sns.set_theme(style="whitegrid")
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    _STYLE_APPLIED = True


def save_figure(
    fig: plt.Figure,
    prefix: str,
    formats: Iterable[str],
    *,
    dpi: int = PNG_DPI,
    close: bool = True,
) -> List[Path]:
    """Save ``fig`` as ``{prefix}.{ext}`` for each requested vector/raster format.

    Unknown formats are ignored. Returns the list of paths written.
    """
    written: List[Path] = []
    for ext in dict.fromkeys(formats):  # de-dupe, preserve order
        if ext not in VALID_FORMATS:
            logger.warning("Ignoring unsupported format: %s", ext)
            continue
        out = Path(f"{prefix}.{ext}")
        fig.savefig(out, dpi=dpi if ext == "png" else None)
        logger.info("Saved %s", out)
        written.append(out)
    if close:
        plt.close(fig)
    return written
