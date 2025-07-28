
#!/usr/bin/env python3
"""
csv_plotter.py – publication‑quality visualisation of qlog‑derived CSVs
(one figure with up to three stacked sub‑plots, optional metadata title).

Metrics:
  - data_rate : receiver goodput & sender throughput (auto‑scaled units)
  - cwnd      : CWND (left) & Send Window (right) (KB/MB/GB) or left if alone
  - rtt       : RTT ± var (left) & Retransmissions (right) or left if alone

Usage:
    python csv_plotter.py [--title] [--formats png pdf svg] \
        [-m receiver-goodput cwnd ...] file.csv
"""
from __future__ import annotations
import argparse
import logging
import math
from pathlib import Path
from typing import Final, Iterable, Optional, Set

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.patches import Patch

# ----------------------------------------------------------------------------#
# Global styling                                                              #
# ----------------------------------------------------------------------------#
plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})
sns.set(style="whitegrid")

# ----------------------------------------------------------------------------#
# Constants & mappings                                                         #
# ----------------------------------------------------------------------------#
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
    "cwnd":    "CWND",
    "swnd":    "Send Window",
    "retx":    "Retransmissions",
    "rtt":     "RTT (ms)",
    "rtt_var": "RTT ± var",
}

_FIGURES: Final[list[tuple[tuple[str, ...], str]]] = [
    (("gp_rcv", "tp_snd"),       "data_rate"),
    (("cwnd", "swnd"),           "cwnd"),
    (("rtt", "rtt_var", "retx"), "rtt"),
]


def _has_data(df: pd.DataFrame, col: str) -> bool:
    return col in df and df[col].notna().any()


def _auto_scale(value: float, units: list[tuple[float, str]]) -> tuple[float, str]:
    for thresh, unit in units:
        if value >= thresh:
            return value / thresh, unit
    return value, units[-1][1]


class CSVPlotter:
    _AX_STEP: Final[int] = 60

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
        self.show_title = title
        fmts = [f for f in out_fmt if f in VALID_FMT]
        self.fmts = tuple(dict.fromkeys(fmts)) or DEFAULT_FMT
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
                    mset.update({"rtt", "rtt_var"})
                else:
                    mset.add(key)
            self.metrics = mset

    def plot(self) -> None:
        df = pd.read_csv(self.path)
        self.log.debug("Loaded %d rows from %s", len(df), self.path.name)

        # convert RTT µs→ms
        if _has_data(df, _COLS["rtt"]):
            df["rtt_ms"] = df[_COLS["rtt"]] / 1_000.0
        if _has_data(df, _COLS["rtt_var"]):
            df["rtt_var_ms"] = df[_COLS["rtt_var"]] / 1_000.0

        # mid‑time & interval
        tcol, t_end = _COLS["t"], _COLS["t_end"]
        if tcol in df and t_end in df:
            df["mid_t"] = (df[tcol] + df[t_end]) / 2.0
            df["interval"] = df[t_end] - df[tcol]
        else:
            df["mid_t"] = df[tcol]
            df["interval"] = 1.0

        # assemble title
        meta = df.iloc[0] if not df.empty else pd.Series(dtype=str)
        ts = meta.get("timestamp (CEST)", "")
        raw_bytes = float(meta.get("test_start.bytes", 0))
        fsz, funit = _auto_scale(
            raw_bytes, [(1e9, "GB"), (1e6, "MB"), (1e3, "KB"), (1, "B")])
        ver = meta.get("version", "")
        serv = meta.get("remote_host", "")
        cli = meta.get("local_host", "")
        port = int(meta.get("remote_port", 0))
        sender = "server" if int(
            meta.get("test_start.reverse", "0")) == 1 else "client"
        proto = meta.get("test_start.protocol", "")
        snd_raw = int(meta.get("sndbuf_actual", 0))
        rcv_raw = int(meta.get("rcvbuf_actual", 0))
        snd_val, snd_unit = _auto_scale(
            snd_raw, [(1e6, "MB"), (1e3, "KB"), (1, "B")])
        rcv_val, rcv_unit = _auto_scale(
            rcv_raw, [(1e6, "MB"), (1e3, "KB"), (1, "B")])
        mss = int(meta.get("tcp_mss_default", 0))
        line1 = f"Time: {ts}    File: {fsz:.2f}{funit}    version: {ver}"
        line2 = f"Server: {serv}:{port}    Client: {cli}    Sender: {sender}"
        line3 = f"Protocol: {proto}    SendBuf: {snd_val:.2f}{snd_unit}    RecvBuf: {rcv_val:.2f}{rcv_unit}    MSS: {mss}"
        title_text = "\n".join((line1, line2, line3))

        # choose panels
        if self.metrics is None:
            blocks = [(k, s) for k, s in _FIGURES if any(
                _has_data(df, _COLS[x]) for x in k)]
        else:
            blocks = []
            for keys, suf in _FIGURES:
                chosen = tuple(
                    x for x in keys if x in self.metrics and _has_data(df, _COLS[x]))
                if chosen:
                    blocks.append((chosen, suf))

        if not blocks:
            self.log.info("No data to plot in %s – skipping.", self.path.name)
            return

        # create figure & axes
        fig, axes = plt.subplots(
            nrows=len(blocks), ncols=1,
            figsize=(20, 6*len(blocks)),
            squeeze=False,
            constrained_layout=True
        )
        axes = axes.flatten()

        # title if requested
        if self.show_title:
            fig.suptitle(title_text, fontsize=16)
            fig.subplots_adjust(top=0.85)

        # plot each
        for ax, (keys, suf) in zip(axes, blocks):
            if suf == "data_rate":
                self._plot_data_rate(df, ax, keys)
            elif suf == "cwnd":
                self._plot_cwnd(df, ax, keys)
            else:
                self._plot_rtt(df, ax, keys)

        # x‑axis ticks & padding
        t0, t1 = df[tcol].min(), df[t_end].max(
        ) if t_end in df else df[tcol].max()
        dur = t1 - t0
        step = 1 if dur <= 10 else 10
        ticks = np.arange(math.floor(t0), math.ceil(t1)+step, step)
        for ax in axes:
            ax.set_xticks(ticks)
            ax.set_xlim(t0 - 0.01*dur, t1 + 0.01*dur)
            ax.set_xlabel("Time (s)", color="black")
            ax.tick_params(axis="x", colors="black")
            ax.grid(True, axis="x")

        # save
        for fmt in self.fmts:
            out = self.path.with_name(f"{self.path.stem}.plots.{fmt}")
            fig.savefig(out, dpi=PNG_DPI if fmt == "png" else None)
            self.log.info("Saved %s", out.name)
        plt.close(fig)

    def _plot_data_rate(self, df: pd.DataFrame, ax: Axes, keys: tuple[str, ...]) -> None:
        raw_max = max(df[_COLS[k]].max()
                      for k in keys if _has_data(df, _COLS[k]))
        if raw_max >= 1e9:
            scale, unit = 1e9, "Gb/s"
        elif raw_max >= 1e6:
            scale, unit = 1e6, "Mb/s"
        elif raw_max >= 1e3:
            scale, unit = 1e3, "Kb/s"
        else:
            scale, unit = 1.0, "b/s"
        for k, color in zip(keys, ("tab:blue", "tab:orange")):
            if _has_data(df, _COLS[k]):
                sns.lineplot(
                    x=df[_COLS["t"]], y=df[_COLS[k]]/scale,
                    ax=ax, label=_LABELS[k],
                    linewidth=1.5, color=color, legend=False
                )
        ax.set_ylabel(f"Data rate ({unit})")
        ax.legend(loc="upper left", frameon=True, framealpha=0.8)

    def _plot_cwnd(self, df: pd.DataFrame, ax_left: Axes, keys: tuple[str, ...]) -> None:
        raw_max = max(df[_COLS[k]].max()
                      for k in keys if _has_data(df, _COLS[k]))
        if raw_max >= 1e6:
            scale, unit = 1e6, "MB"
        elif raw_max >= 1e3:
            scale, unit = 1e3, "KB"
        else:
            scale, unit = 1.0, "B"
        handles, labels = [], []
        # CWND
        if "cwnd" in keys and _has_data(df, _COLS["cwnd"]):
            h = sns.lineplot(
                x=df[_COLS["t"]], y=df[_COLS["cwnd"]]/scale,
                ax=ax_left, linewidth=1.5, color="tab:green",
                label=_LABELS["cwnd"], legend=False
            )
            handles.append(h.lines[0])
            labels.append(_LABELS["cwnd"])
            ax_left.set_ylabel(f"CWND ({unit})", color="tab:green")
            ax_left.tick_params(labelcolor="tab:green")
        # Send Window
        if "swnd" in keys and _has_data(df, _COLS["swnd"]):
            if "cwnd" in keys:
                ax2 = ax_left.twinx()
                ax2.spines["right"].set_position(("axes", 1.0))
            else:
                ax2 = ax_left
            h2 = sns.lineplot(
                x=df[_COLS["t"]], y=df[_COLS["swnd"]]/scale,
                ax=ax2, linewidth=1.5, color="tab:red",
                label=_LABELS["swnd"], legend=False
            )
            handles.append(h2.lines[0])
            labels.append(_LABELS["swnd"])
            ax2.set_ylabel(f"Send Window ({unit})", color="tab:red")
            ax2.tick_params(labelcolor="tab:red")
            if ax2 is not ax_left:
                ax2.grid(False)
        ax_left.legend(handles, labels, loc="upper left",
                       frameon=True, framealpha=0.8)

    def _plot_rtt(self, df: pd.DataFrame, ax: Axes, keys: tuple[str, ...]) -> None:
        handles, labels = [], []
        has_rtt = "rtt" in keys and _has_data(df, _COLS["rtt"])
        has_retx = "retx" in keys and _has_data(df, _COLS["retx"])
        # Retransmissions
        if has_retx:
            if has_rtt:
                ax2 = ax.twinx()
                ax2.spines["right"].set_position(("axes", 1.0))
            else:
                ax2 = ax
            ax2.bar(
                df[_COLS["t"]], df[_COLS["retx"]],
                width=df["interval"], align="edge",
                alpha=0.3, color="tab:grey", zorder=1
            )
            handles.append(Patch(facecolor="tab:grey", alpha=0.3))
            labels.append(_LABELS["retx"])
            ax2.set_ylabel("Retransmissions", color="tab:grey")
            ax2.tick_params(labelcolor="tab:grey")
            if ax2 is not ax:
                ax2.grid(False)
        # RTT ± var & line
        if has_rtt:
            if "rtt_var" in keys and _has_data(df, _COLS["rtt_var"]):
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
                x="mid_t", y="rtt_ms", data=df, ax=ax,
                linewidth=1.5, color="tab:purple",
                label=_LABELS["rtt"], zorder=3, legend=False
            )
            handles.append(line.lines[0])
            labels.append(_LABELS["rtt"])
            ax.set_ylabel("RTT (ms)", color="tab:purple")
            ax.tick_params(labelcolor="tab:purple")
        # legend
        ax.legend(handles, labels, loc="upper left",
                  frameon=True, framealpha=0.8)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot qlog/QoE metrics from CSV.")
    p.add_argument("files", nargs="+", type=Path, help="Input CSV file(s)")
    p.add_argument("--title", action="store_true",
                   help="Show three-line metadata title")
    p.add_argument("--formats", nargs="+", choices=sorted(VALID_FMT),
                   metavar="FMT", help="Output formats (png, pdf, svg)")
    p.add_argument("-m", "--metrics", nargs="+", choices=sorted(_METRIC_MAP.keys()),
                   metavar="METRIC", help="Only plot these metrics")
    p.add_argument("-v", "--verbose", action="count", default=0,
                   help="-v for INFO, -vv for DEBUG")
    return p


def _configure_logging(verbosity: int) -> None:
    level = logging.INFO if verbosity <= 0 else logging.DEBUG
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.ERROR)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    _configure_logging(args.verbose)
    for f in args.files:
        try:
            CSVPlotter(
                f,
                title=args.title,
                out_fmt=args.formats or DEFAULT_FMT,
                metrics=args.metrics
            ).plot()
        except Exception:
            logging.exception("Failed to plot %s", f)


if __name__ == "__main__":
    main()
