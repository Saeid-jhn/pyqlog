#!/usr/bin/env python3
"""
csv_plotter.py – publication‑quality visualisation of qlog‑derived CSVs
(one combined figure with three stacked sub‑plots).

OUTPUT
------
<stem>.plots.<fmt>
    ├─ top : data‑rate       (rcv_goodput / snd_throughput)
    ├─ mid : cwnd            (snd_cwnd left, snd_wnd right)
    └─ bot : RTT             (RTT line ± variance, retransmissions bars)

CLI EXAMPLE
-----------
python csv_plotter.py run.csv --add-title --formats png pdf -v
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Final, Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------#
# Global appearance                                                           #
# ---------------------------------------------------------------------------#
plt.rcParams.update(
    {
        "font.size": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
    }
)
sns.set(style="whitegrid")

# ---------------------------------------------------------------------------#
# Constants & column map                                                      #
# ---------------------------------------------------------------------------#
PNG_DPI: Final[int] = 600
DEFAULT_FMT: Final[tuple[str, ...]] = ("png",)
VALID_FMT: Final[frozenset[str]] = frozenset(("png", "pdf", "svg"))

COLS: Final[dict[str, str]] = {
    "t":        "start_time (s)",
    "gp_rcv":   "rcv_goodput (bps)",
    "tp_snd":   "snd_throughput (bps)",
    "retx":     "retransmits",
    "cwnd":     "snd_cwnd (K)",
    "swnd":     "snd_wnd (K)",
    "rtt":      "rtt (us)",
    "rtt_var":  "rtt_var (us)",
}

FIGURES: Final[list[tuple[tuple[str, ...], str]]] = [
    (("gp_rcv", "tp_snd"), "data_rate"),
    (("cwnd", "swnd"), "cwnd"),
    (("rtt", "rtt_var", "retx"), "rtt"),
]

# ---------------------------------------------------------------------------#
# Helpers                                                                     #
# ---------------------------------------------------------------------------#


def _has_data(df: pd.DataFrame, col: str) -> bool:
    """True if *col* exists & has ≥1 non‑NaN value."""
    return col in df and df[col].notna().any()


# ---------------------------------------------------------------------------#
# Core plotter                                                                #
# ---------------------------------------------------------------------------#


class CSVPlotter:
    """Render up to three sub‑plots in one combined figure."""

    _AX_STEP: Final[int] = 60  # secondary‑axis offset (cwnd & rtt plots)

    def __init__(
        self,
        csv_path: Path,
        *,
        title: bool = False,
        out_fmt: Iterable[str] = DEFAULT_FMT,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.path = csv_path
        self.title = title
        self.fmts = tuple(
            fmt for fmt in out_fmt if fmt in VALID_FMT) or DEFAULT_FMT
        self.log = logger or logging.getLogger("CSVPlotter")

    # ------------------------------------------------------------------ #
    # Public entry                                                        #
    # ------------------------------------------------------------------ #
    def plot(self) -> None:
        df = pd.read_csv(self.path)
        self.log.debug("Loaded %d rows from %s", len(df), self.path.name)

        # Pre‑compute RTT in ms & variance band
        if _has_data(df, COLS["rtt"]):
            df["rtt_ms"] = df[COLS["rtt"]] / 1_000.0
        if _has_data(df, COLS["rtt_var"]):
            df["rtt_var_ms"] = df[COLS["rtt_var"]] / 1_000.0

        # Decide which figure blocks actually have something to show
        blocks: list[tuple[tuple[str, ...], str]] = [
            (keys, suffix)
            for keys, suffix in FIGURES
            if any(_has_data(df, COLS[k]) for k in keys)
        ]
        if not blocks:
            self.log.info(
                "No usable metrics – nothing plotted for %s", self.path)
            return

        # ----------------------------------------------------------------
        # Create combined figure & sub‑plots
        # ----------------------------------------------------------------
        fig, axes = plt.subplots(
            nrows=len(blocks),
            ncols=1,
            figsize=(20, 6 * len(blocks)),
            sharex=False,
            squeeze=False,
        )
        axes = axes.flatten()

        if self.title:
            fig.suptitle(self.path.stem.replace(
                "_", " "), y=0.995, fontsize=14)

        for ax, (keys, suffix) in zip(axes, blocks):
            if suffix == "data_rate":
                self._plot_data_rate(df, ax, keys)
            elif suffix == "cwnd":
                self._plot_cwnd(df, ax, keys)
            elif suffix == "rtt":
                self._plot_rtt(df, ax, keys)

        # Consistent x‑axis ticks/grid across all sub‑plots
        bottom_ax = axes[-1]
        if len(df):
            t_max = df[COLS["t"]].max()
            ticks = np.arange(0, t_max + 10, 10)
            bottom_ax.set_xticks(ticks)
        bottom_ax.set_xlim(left=0)
        bottom_ax.set_xlabel("Time (s)")
        for ax in axes:
            ax.grid(True, axis="x")

        fig.tight_layout(rect=(0, 0, 1, 0.97))

        for fmt in self.fmts:
            outfile = self.path.with_name(f"{self.path.stem}.plots.{fmt}")
            fig.savefig(outfile, dpi=PNG_DPI if fmt == "png" else None)
            self.log.info("Saved %s", outfile.name)

        plt.close(fig)

    # ------------------------------------------------------------------ #
    # Plot helpers                                                       #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _plot_data_rate(df: pd.DataFrame, ax: Axes, keys: tuple[str, ...]) -> None:
        colour_cycle = ("tab:blue", "tab:orange")
        for k, colour in zip(keys, colour_cycle, strict=False):
            if _has_data(df, COLS[k]):
                sns.lineplot(
                    x=COLS["t"],
                    y=COLS[k],
                    data=df,
                    ax=ax,
                    label=COLS[k],
                    linewidth=1.5,
                    color=colour,
                )
        ax.set_ylabel("Data rate (bps)")
        ax.legend(loc="upper left")

    def _plot_cwnd(self, df: pd.DataFrame, ax_left: Axes, keys: tuple[str, ...]) -> None:
        """cwnd left axis (green); send‑window right axis (red)."""
        handles, labels = [], []

        # CWND on left
        if "cwnd" in keys and _has_data(df, COLS["cwnd"]):
            h = sns.lineplot(
                x=COLS["t"],
                y=COLS["cwnd"],
                data=df,
                ax=ax_left,
                linewidth=1.5,
                color="tab:green",
                label=COLS["cwnd"],
            )
            handles.append(h.lines[0])
            labels.append(COLS["cwnd"])
            ax_left.set_ylabel("CWND (K)", color="tab:green")
            ax_left.tick_params(axis="y", labelcolor="tab:green")

        # Send window on right if present
        if "swnd" in keys and _has_data(df, COLS["swnd"]):
            ax_right = ax_left.twinx()
            ax_right.spines["right"].set_position(("outward", self._AX_STEP))
            h2 = sns.lineplot(
                x=COLS["t"],
                y=COLS["swnd"],
                data=df,
                ax=ax_right,
                linewidth=1.5,
                color="tab:red",
                label=COLS["swnd"],
            )
            handles.append(h2.lines[0])
            labels.append(COLS["swnd"])
            ax_right.set_ylabel("Send window (K)", color="tab:red")
            ax_right.tick_params(axis="y", labelcolor="tab:red")
            ax_right.grid(False)

        ax_left.legend(handles, labels, loc="upper left")

    def _plot_rtt(self, df: pd.DataFrame, ax: Axes, keys: tuple[str, ...]) -> None:
        """Draw retransmissions bars first, then RTT ±variance band & line."""
        handles, labels = [], []

        # Retransmissions on right‑hand axis (drawn FIRST so RTT overlays)
        if "retx" in keys and _has_data(df, COLS["retx"]):
            ax2 = ax.twinx()
            ax2.spines["right"].set_position(("outward", self._AX_STEP))
            bars = sns.barplot(
                x=COLS["t"],
                y=COLS["retx"],
                data=df,
                ax=ax2,
                alpha=0.3,
                color="tab:grey",
                zorder=1,
            )
            handles.append(Patch(facecolor="tab:grey", alpha=0.3))
            labels.append("Retransmissions")
            ax2.set_ylabel("Retransmissions", color="tab:grey")
            ax2.tick_params(axis="y", labelcolor="tab:grey")
            ax2.grid(False)

        # RTT ± variance (primary axis)
        if "rtt" in keys and _has_data(df, "rtt_ms"):
            # Variance band first
            if "rtt_var" in keys and _has_data(df, "rtt_var_ms"):
                lower = (df["rtt_ms"] - df["rtt_var_ms"]).clip(lower=0)
                upper = df["rtt_ms"] + df["rtt_var_ms"]
                band = ax.fill_between(
                    df[COLS["t"]],
                    lower,
                    upper,
                    alpha=0.25,
                    color="tab:purple",
                    label="RTT ± var",
                    zorder=2,
                )
                handles.append(band)
                labels.append("RTT ± var")
            # RTT line on top
            line = sns.lineplot(
                x=COLS["t"],
                y="rtt_ms",
                data=df,
                ax=ax,
                linewidth=1.5,
                color="tab:purple",
                label="RTT (ms)",
                zorder=3,
            )
            handles.append(line.lines[0])
            labels.append("RTT (ms)")

        ax.set_ylabel("RTT (ms)", color="tab:purple")
        ax.tick_params(axis="y", labelcolor="tab:purple")
        ax.legend(handles, labels, loc="upper left")

    # ------------------------------------------------------------------ #
    # Utility                                                            #
    # ------------------------------------------------------------------ #


def plot_csv(file: str | Path) -> None:
    """Plot one CSV (combined figure)."""
    _configure_logging(0)
    CSVPlotter(Path(file)).plot()


# ---------------------------------------------------------------------------#
# CLI                                                                        #
# ---------------------------------------------------------------------------#


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot qlog/QoE metrics from CSV.")
    p.add_argument("files", nargs="+", type=Path)
    p.add_argument("--add-title", action="store_true")
    p.add_argument(
        "--formats",
        nargs="+",
        choices=sorted(VALID_FMT),
        metavar="FMT",
        help="png pdf svg (default: png)",
    )
    p.add_argument(
        "-v", "--verbose", action="count", default=0, help="-v INFO, -vv DEBUG"
    )
    return p


def _configure_logging(verbosity: int) -> None:
    logging.basicConfig(
        level=logging.INFO if verbosity == 0 else logging.DEBUG,
        format="%(levelname)s %(message)s",
    )
    logging.getLogger("matplotlib").setLevel(logging.ERROR)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    _configure_logging(args.verbose)
    for f in args.files:
        try:
            CSVPlotter(
                f,
                title=args.add_title,
                out_fmt=args.formats or DEFAULT_FMT,
            ).plot()
        except Exception:
            logging.exception("Failed %s", f)


if __name__ == "__main__":
    main()
