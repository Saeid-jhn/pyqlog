#!/usr/bin/env python3
"""
csv_plotter.py – plot qlog‑derived CSVs (goodput, retrans, cwnd, RTT).

Default: PNG @ 600 dpi.
Extra formats: pdf / svg with --formats.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Final, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

# --- style -----------------------------------------------------------------
plt.rcParams.update(
    {"font.size": 12, "axes.labelsize": 12, "xtick.labelsize": 12,
     "ytick.labelsize": 12, "legend.fontsize": 12}
)
sns.set(style="whitegrid")

# --- constants -------------------------------------------------------------
PNG_DPI: Final[int] = 600
DEFAULT_FMT: Final[Tuple[str, ...]] = ("png",)
VALID_FMT: Final[set[str]] = {"png", "pdf", "svg"}

COLS = {
    "time": "start time (sec)",
    "goodput": "goodput (bits/sec)",
    "retrans": "Retransmissions",
    "cwnd": "cwnd (K)",
    "rtt_us": "RTT (microsecond)",
    "rtt_ms": "RTT (ms)",
}


# --- plotter ---------------------------------------------------------------
class CSVPlotter:
    _AX_STEP: Final[int] = 60  # pts between extra y‑axes

    def __init__(
        self,
        csv_path: Path,
        *,
        title: bool = False,
        out_fmt: Tuple[str, ...] = DEFAULT_FMT,
    ):
        self.p = csv_path
        self.title = title
        self.fmts = tuple(f for f in out_fmt if f in VALID_FMT) or DEFAULT_FMT
        self.log = logging.getLogger("CSVPlotter")

    def plot(self) -> None:
        df = pd.read_csv(self.p)
        if COLS["rtt_us"] in df and COLS["rtt_ms"] not in df:
            df[COLS["rtt_ms"]] = df[COLS["rtt_us"]] / 1_000.0

        fig, ax1 = plt.subplots(figsize=(25, 6))
        if self.title:
            ax1.set_title(self.p.stem)

        # goodput
        sns.lineplot(x=COLS["time"], y=COLS["goodput"], data=df,
                     ax=ax1, color="tab:blue", label="Goodput")
        ax1.set_xlabel("Time (sec)")
        ax1.set_ylabel("Goodput (bits/sec)", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")

        lines, labels = ax1.get_legend_handles_labels()
        if ax1.get_legend():
            ax1.get_legend().remove()

        off = self._AX_STEP

        # retrans
        if COLS["retrans"] in df:
            ax2 = ax1.twinx()
            ax2.spines["right"].set_position(("outward", off))
            sns.barplot(x=COLS["time"], y=COLS["retrans"], data=df,
                        ax=ax2, color="tab:red", alpha=0.3)
            ax2.set_ylabel("Retransmissions", color="tab:red")
            ax2.tick_params(axis="y", labelcolor="tab:red")
            ax2.grid(False)
            lines.append(Patch(facecolor="tab:red",
                         alpha=0.3, label="Retransmissions"))
            labels.append("Retransmissions")
            off += self._AX_STEP

        # cwnd
        if COLS["cwnd"] in df:
            ax3 = ax1.twinx()
            ax3.spines["right"].set_position(("outward", off))
            sns.lineplot(x=COLS["time"], y=COLS["cwnd"], data=df,
                         ax=ax3, color="tab:green", label="cwnd (K)")
            ax3.set_ylabel("cwnd (K)", color="tab:green")
            ax3.tick_params(axis="y", labelcolor="tab:green")
            ax3.grid(False)
            hl, ll = ax3.get_legend_handles_labels()
            lines += hl
            labels += ll
            if ax3.get_legend():
                ax3.get_legend().remove()
            off += self._AX_STEP

        # rtt
        if COLS["rtt_ms"] in df:
            ax4 = ax1.twinx()
            ax4.spines["right"].set_position(("outward", off))
            sns.lineplot(x=COLS["time"], y=COLS["rtt_ms"], data=df,
                         ax=ax4, color="tab:purple", linestyle="--", label="RTT (ms)")
            ax4.set_ylabel("RTT (ms)", color="tab:purple")
            ax4.tick_params(axis="y", labelcolor="tab:purple")
            ax4.grid(False)
            hl, ll = ax4.get_legend_handles_labels()
            lines += hl
            labels += ll
            if ax4.get_legend():
                ax4.get_legend().remove()

        # dedup legend
        uniq = {lbl: ln for ln, lbl in zip(lines, labels)}
        ax1.legend(uniq.values(), uniq.keys(), loc="upper left")

        # x‑axis
        if len(df):
            ax1.set_xticks(np.arange(0, df[COLS["time"]].max() + 10, 10))
        ax1.set_xlim(left=0)
        ax1.grid(True, axis="x")

        fig.tight_layout()

        for fmt in self.fmts:
            fig.savefig(self.p.with_suffix(
                f".{fmt}"), dpi=PNG_DPI if fmt == "png" else None)
        plt.close(fig)


# --- CLI -------------------------------------------------------------------
def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot network metrics CSVs.")
    p.add_argument("files", nargs="+", type=Path)
    p.add_argument("--add-title", action="store_true")
    p.add_argument("--formats", nargs="+", choices=sorted(VALID_FMT),
                   metavar="FMT", help="png pdf svg (default: png)")
    p.add_argument("-v", "--verbose", action="count", default=0,
                   help="-v INFO, -vv DEBUG")
    return p


def _config_log(verb: int) -> None:
    logging.basicConfig(
        level=logging.INFO if verb == 0 else logging.DEBUG,
        format="%(levelname)s %(message)s",
    )
    logging.getLogger("matplotlib").setLevel(logging.ERROR)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parser().parse_args(argv)
    _config_log(args.verbose)
    for f in args.files:
        try:
            CSVPlotter(f, title=args.add_title,
                       out_fmt=tuple(args.formats or DEFAULT_FMT)).plot()
        except Exception:
            logging.exception("Failed %s", f)


if __name__ == "__main__":
    main()
