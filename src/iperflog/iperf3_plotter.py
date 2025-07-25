#!/usr/bin/env python3
"""
csv_plotter.py – publication‑quality visualisation of qlog‑derived CSVs
(one combined figure with up to three stacked sub‑plots), with optional
selection of individual metrics.

Data‑rate is auto‑scaled to bps, Kbps, Mbps, or Gbps based on your trace,
and legends only show metric names (units are on the y‑axis).

OUTPUT
------
<stem>.plots.<fmt>
    ├─ data_rate : receiver goodput and/or sender throughput (auto‑scaled units)
    ├─ cwnd      : CWND (left) and send window (right)
    └─ rtt       : RTT ± variance (left) and retransmissions (right)

By default, all available metrics are plotted. Use `-m/--metrics` to restrict
to a subset. Selecting `rtt` will always include both RTT and its variance.

CLI EXAMPLES
------------
# All metrics (default):
python csv_plotter.py run.csv --formats png pdf

# Only receiver goodput + cwnd:
python csv_plotter.py run.csv -m receiver-goodput cwnd

# Only RTT (with variance) + retransmits:
python csv_plotter.py run.csv -m rtt retransmits
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Final, Iterable, Optional, Sequence, Set

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------#
# Global appearance                                                           #
# ---------------------------------------------------------------------------#
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})
sns.set(style="whitegrid")

# ---------------------------------------------------------------------------#
# Constants & mappings                                                        #
# ---------------------------------------------------------------------------#
PNG_DPI: Final[int] = 300
DEFAULT_FMT: Final[tuple[str, ...]] = ("png",)
VALID_FMT: Final[Set[str]] = frozenset(("png", "pdf", "svg"))

_METRIC_MAP: Final[dict[str, str]] = {
    "receiver-goodput":  "gp_rcv",
    "sender-throughput": "tp_snd",
    "cwnd":              "cwnd",
    "send-window":       "swnd",
    "rtt":               "rtt",
    "rtt-var":           "rtt_var",
    "retransmits":       "retx",
}

_COLS: Final[dict[str, str]] = {
    "t":        "start_time (s)",
    "t_end":    "end_time (s)",
    "gp_rcv":   "rcv_goodput (bps)",
    "tp_snd":   "snd_throughput (bps)",
    "retx":     "retransmits",
    "cwnd":     "snd_cwnd (K)",
    "swnd":     "snd_wnd (K)",
    "rtt":      "rtt (us)",
    "rtt_var":  "rtt_var (us)",
}

_LABELS: Final[dict[str, str]] = {
    "gp_rcv":  "Receiver Goodput",
    "tp_snd":  "Sender Throughput",
    "cwnd":    "CWND (K)",
    "swnd":    "Send Window (K)",
    "retx":    "Retransmissions",
    "rtt":     "RTT (ms)",
    "rtt_var": "RTT ± var",
}

_FIGURES: Final[list[tuple[tuple[str, ...], str]]] = [
    (("gp_rcv", "tp_snd"), "data_rate"),
    (("cwnd", "swnd"), "cwnd"),
    (("rtt", "rtt_var", "retx"), "rtt"),
]


def _has_data(df: pd.DataFrame, col: str) -> bool:
    """True if column exists and has at least one non-NaN."""
    return col in df and df[col].notna().any()


class CSVPlotter:
    """Render selected metric sub‑plots from a qlog‑derived CSV."""

    _AX_STEP: Final[int] = 60  # offset for secondary axes

    def __init__(
        self,
        csv_path: Path,
        *,
        title: bool = False,
        out_fmt: Iterable[str] = DEFAULT_FMT,
        metrics: Optional[Iterable[str]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.path = csv_path
        self.title = title

        fmt_list = [f for f in out_fmt if f in VALID_FMT]
        self.fmts = tuple(dict.fromkeys(fmt_list)) or DEFAULT_FMT

        self.log = logger or logging.getLogger("CSVPlotter")

        if metrics is None:
            self.metrics: Optional[Set[str]] = None
        else:
            mset: Set[str] = set()
            for m in metrics:
                if m not in _METRIC_MAP:
                    continue
                key = _METRIC_MAP[m]
                if m == "rtt":
                    mset.update(("rtt", "rtt_var"))
                else:
                    mset.add(key)
            self.metrics = mset

        self.log.debug(
            "Initialized: %s (title=%s, fmts=%r, metrics=%r)",
            self.path.name, self.title, self.fmts, self.metrics
        )

    def plot(self) -> None:
        df = pd.read_csv(self.path)
        self.log.debug("Loaded %d rows from %s", len(df), self.path.name)

        # convert µs → ms for RTT & variance
        if _has_data(df, _COLS["rtt"]):
            df["rtt_ms"] = df[_COLS["rtt"]] / 1_000.0
        if _has_data(df, _COLS["rtt_var"]):
            df["rtt_var_ms"] = df[_COLS["rtt_var"]] / 1_000.0

        # compute mid-time & interval
        if _COLS["t"] in df and _COLS["t_end"] in df:
            df["mid_t"] = (df[_COLS["t"]] + df[_COLS["t_end"]]) / 2.0
            df["interval"] = df[_COLS["t_end"]] - df[_COLS["t"]]
        else:
            df["mid_t"] = df[_COLS["t"]]
            df["interval"] = 1.0

        # select blocks
        if self.metrics is None:
            blocks = [
                (keys, suffix)
                for keys, suffix in _FIGURES
                if any(_has_data(df, _COLS[k]) for k in keys)
            ]
        else:
            blocks = []
            for keys, suffix in _FIGURES:
                chosen = tuple(
                    k for k in keys
                    if k in self.metrics and _has_data(df, _COLS[k])
                )
                if chosen:
                    blocks.append((chosen, suffix))

        if not blocks:
            self.log.info("No data to plot in %s—skipping.", self.path.name)
            return

        fig, axes = plt.subplots(
            nrows=len(blocks), ncols=1,
            figsize=(20, 6 * len(blocks)), squeeze=False
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

        # uniform x‑axis
        if len(df):
            t0 = df[_COLS["t"]].min()
            t1 = df[_COLS["t_end"]].max(
            ) if _COLS["t_end"] in df else df[_COLS["t"]].max()
            ticks = np.arange(t0, t1 + 10, 10)
            for ax in axes:
                ax.set_xticks(ticks)
                ax.set_xlim(t0, t1)
                ax.grid(True, axis="x")
            axes[-1].set_xlabel("Time (s)")

        fig.tight_layout(rect=(0, 0, 1, 0.97))

        for fmt in self.fmts:
            out = self.path.with_name(f"{self.path.stem}.plots.{fmt}")
            self.log.debug("Saving %s", out.name)
            fig.savefig(out, dpi=PNG_DPI if fmt == "png" else None)
            self.log.info("Saved %s", out.name)

        plt.close(fig)

    def _plot_data_rate(self, df: pd.DataFrame, ax: Axes, keys: tuple[str, ...]) -> None:
        """
        Plot receiver goodput and/or sender throughput,
        auto‑scaling to bps, Kbps, Mbps, or Gbps.
        """
        raw_max = max(
            (df[_COLS[k]].max() for k in keys if _has_data(df, _COLS[k])),
            default=0.0
        )
        if raw_max >= 1e9:
            scale, unit = 1e9, "Gb/s"
        elif raw_max >= 1e6:
            scale, unit = 1e6, "Mb/s"
        elif raw_max >= 1e3:
            scale, unit = 1e3, "Kb/s"
        else:
            scale, unit = 1.0, "b/s"

        colours = ("tab:blue", "tab:orange")
        handles, labels = [], []
        for k, c in zip(keys, colours, strict=False):
            if _has_data(df, _COLS[k]):
                line = sns.lineplot(
                    x=df[_COLS["t"]],
                    y=df[_COLS[k]] / scale,
                    ax=ax,
                    label=_LABELS[k],
                    linewidth=1.5,
                    color=c,
                    legend=False,
                )
                handles.append(line.lines[0])
                labels.append(_LABELS[k])

        ax.set_ylabel(f"Data rate ({unit})")
        ax.legend(handles, labels, loc="upper left",
                  frameon=True, framealpha=0.8)

    def _plot_cwnd(self, df: pd.DataFrame, ax_left: Axes, keys: tuple[str, ...]) -> None:
        """Plot CWND (left) and send‑window (right)."""
        handles, labels = [], []

        if "cwnd" in keys and _has_data(df, _COLS["cwnd"]):
            h = sns.lineplot(
                x=df[_COLS["t"]], y=df[_COLS["cwnd"]],
                ax=ax_left, linewidth=1.5, color="tab:green",
                label=_LABELS["cwnd"], legend=False
            )
            handles.append(h.lines[0])
            labels.append(_LABELS["cwnd"])
            ax_left.set_ylabel("CWND (K)", color="tab:green")
            ax_left.tick_params(axis="y", labelcolor="tab:green")

        if "swnd" in keys and _has_data(df, _COLS["swnd"]):
            ax_right = ax_left.twinx()
            ax_right.spines["right"].set_position(("outward", self._AX_STEP))
            h2 = sns.lineplot(
                x=df[_COLS["t"]], y=df[_COLS["swnd"]],
                ax=ax_right, linewidth=1.5, color="tab:red",
                label=_LABELS["swnd"], legend=False
            )
            handles.append(h2.lines[0])
            labels.append(_LABELS["swnd"])
            ax_right.set_ylabel("Send Window (K)", color="tab:red")
            ax_right.tick_params(axis="y", labelcolor="tab:red")
            ax_right.grid(False)

        ax_left.legend(handles, labels, loc="upper left",
                       frameon=True, framealpha=0.8)

    def _plot_rtt(self, df: pd.DataFrame, ax: Axes, keys: tuple[str, ...]) -> None:
        """Plot RTT ± variance and retransmissions."""
        handles, labels = [], []

        # retransmissions bars on right axis
        if "retx" in keys and _has_data(df, _COLS["retx"]):
            ax2 = ax.twinx()
            ax2.spines["right"].set_position(("outward", self._AX_STEP))
            ax2.bar(
                df[_COLS["t"]], df[_COLS["retx"]],
                width=df["interval"], align="edge",
                alpha=0.3, color="tab:grey", zorder=1
            )
            handles.append(Patch(facecolor="tab:grey", alpha=0.3))
            labels.append(_LABELS["retx"])
            ax2.set_ylabel("Retransmissions", color="tab:grey")
            ax2.tick_params(axis="y", labelcolor="tab:grey")
            ax2.grid(False)

        # RTT ± variance band then RTT line on left axis
        if "rtt" in keys and _has_data(df, "rtt_ms"):
            if "rtt_var" in keys and _has_data(df, "rtt_var_ms"):
                lower = (df["rtt_ms"] - df["rtt_var_ms"]).clip(lower=0)
                upper = df["rtt_ms"] + df["rtt_var_ms"]
                band = ax.fill_between(
                    df["mid_t"], lower, upper,
                    alpha=0.25, color="tab:purple",
                    label=_LABELS["rtt_var"], zorder=2
                )
                handles.append(band)
                labels.append(_LABELS["rtt_var"])

            line = sns.lineplot(
                x="mid_t", y="rtt_ms", data=df,
                ax=ax, linewidth=1.5, color="tab:purple",
                label=_LABELS["rtt"], zorder=3, legend=False
            )
            handles.append(line.lines[0])
            labels.append(_LABELS["rtt"])

        ax.set_ylabel("RTT (ms)", color="tab:purple")
        ax.tick_params(axis="y", labelcolor="tab:purple")
        ax.legend(handles, labels, loc="upper left",
                  frameon=True, framealpha=0.8)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot qlog/QoE metrics from CSV.")
    p.add_argument("files", nargs="+", type=Path, help="Input CSV file(s)")
    p.add_argument(
        "--add-title", action="store_true",
        help="Add filename stem as the figure title"
    )
    p.add_argument(
        "--formats", nargs="+", choices=sorted(VALID_FMT), metavar="FMT",
        help="Output formats (png, pdf, svg). Default: png"
    )
    p.add_argument(
        "-m", "--metrics", nargs="+", choices=sorted(_METRIC_MAP.keys()),
        metavar="METRIC",
        help=(
            "Only plot these metrics (default=all). "
            "Choices: " + ", ".join(sorted(_METRIC_MAP.keys()))
        )
    )
    p.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="-v for INFO, -vv for DEBUG"
    )
    return p


def _configure_logging(verbosity: int) -> None:
    level = logging.INFO if verbosity <= 1 else logging.DEBUG
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")
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
                metrics=args.metrics,
            ).plot()
        except Exception:
            logging.exception("Failed to plot %s", f)


if __name__ == "__main__":
    main()
