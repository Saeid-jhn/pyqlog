"""Publication-quality iperf3 plots from an :class:`AnalysisResult`.

One figure with up to three stacked panels (data rate, CWND/send-window,
RTT/retransmits). Each panel is drawn only when its data is present. An optional
three-line metadata title summarizes the run.
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Dict, Final, Iterable, List, Optional, Set

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.patches import Patch

from ..core import AnalysisResult, apply_style, auto_scale, save_figure
from ..core.units import BYTES
from .analyzer import INTERVAL_COLS

logger = logging.getLogger(__name__)

# User-facing metric name -> internal key.
METRIC_MAP: Final[Dict[str, str]] = {
    "receiver-goodput": "gp_rcv",
    "sender-throughput": "tp_snd",
    "cwnd": "cwnd",
    "send-window": "swnd",
    "rtt": "rtt",
    "rtt-var": "rtt_var",
    "retransmits": "retx",
}

# Internal key -> interval column name.
_COLS: Final[Dict[str, str]] = {
    "t": "start_time (s)",
    "t_end": "end_time (s)",
    "gp_rcv": "rcv_goodput (bps)",
    "tp_snd": "snd_throughput (bps)",
    "retx": "retransmits",
    "cwnd": "snd_cwnd (K)",
    "swnd": "snd_wnd (K)",
    "rtt": "rtt (us)",
    "rtt_var": "rtt_var (us)",
}

_LABELS: Final[Dict[str, str]] = {
    "gp_rcv": "Receiver Goodput",
    "tp_snd": "Sender Throughput",
    "cwnd": "CWND",
    "swnd": "Send Window",
    "retx": "Retransmissions",
    "rtt": "RTT (ms)",
    "rtt_var": "RTT ± var",
}

# (internal keys in panel, panel id) in draw order.
_FIGURES: Final[list] = [
    (("gp_rcv", "tp_snd"), "data_rate"),
    (("cwnd", "swnd"), "cwnd"),
    (("rtt", "rtt_var", "retx"), "rtt"),
]

_DATA_RATE_UNITS = [(1e9, "Gb/s"), (1e6, "Mb/s"), (1e3, "Kb/s"), (1.0, "b/s")]


def _has_data(df: pd.DataFrame, col: str) -> bool:
    return col in df and df[col].notna().any()


class IperfPlotter:
    """Plot an iperf :class:`AnalysisResult` to one or more image formats."""

    def __init__(self, result: AnalysisResult, *, title: bool = False,
                 metrics: Optional[Iterable[str]] = None):
        if result.kind != "iperf":
            raise ValueError(f"IperfPlotter needs an iperf result, got {result.kind!r}")
        self.result = result
        self.show_title = title
        self.metrics = self._resolve_metrics(metrics)

    @staticmethod
    def _resolve_metrics(metrics: Optional[Iterable[str]]) -> Optional[Set[str]]:
        if metrics is None:
            return None
        chosen: Set[str] = set()
        for m in metrics:
            if m not in METRIC_MAP:
                continue
            if m == "rtt":
                chosen.update({"rtt", "rtt_var"})
            else:
                chosen.add(METRIC_MAP[m])
        return chosen

    def render(
        self,
        *,
        out_dir: Optional[str] = None,
        formats: Iterable[str] = ("png",),
        prefix: Optional[str] = None,
    ) -> List[Path]:
        apply_style()
        df = self.result.table("intervals").copy()
        if df.empty:
            logger.info("No interval data for %s", self.result.source)
            return []

        # RTT µs -> ms.
        if _has_data(df, _COLS["rtt"]):
            df["rtt_ms"] = df[_COLS["rtt"]] / 1_000.0
        if _has_data(df, _COLS["rtt_var"]):
            df["rtt_var_ms"] = df[_COLS["rtt_var"]] / 1_000.0

        tcol, t_end = _COLS["t"], _COLS["t_end"]
        if tcol in df and t_end in df:
            df["mid_t"] = (df[tcol] + df[t_end]) / 2.0
            df["interval"] = df[t_end] - df[tcol]
        else:
            df["mid_t"] = df[tcol]
            df["interval"] = 1.0

        blocks = self._select_blocks(df)
        if not blocks:
            logger.info("Nothing to plot for %s", self.result.source)
            return []

        fig, axes = plt.subplots(nrows=len(blocks), ncols=1,
                                 figsize=(20, 6 * len(blocks)),
                                 squeeze=False, constrained_layout=True)
        axes = axes.flatten()

        if self.show_title:
            # constrained_layout handles spacing; suptitle alone is enough.
            fig.suptitle(self._title_text(), fontsize=16)

        for ax, (keys, panel) in zip(axes, blocks):
            if panel == "data_rate":
                self._plot_data_rate(df, ax, keys)
            elif panel == "cwnd":
                self._plot_cwnd(df, ax, keys)
            else:
                self._plot_rtt(df, ax, keys)

        self._format_xaxis(df, axes, tcol, t_end)

        prefix = self._resolve_prefix(prefix, out_dir)
        return save_figure(fig, prefix, formats)

    # -- selection / layout ---------------------------------------------- #

    def _select_blocks(self, df: pd.DataFrame) -> list:
        blocks = []
        for keys, panel in _FIGURES:
            if self.metrics is None:
                chosen = tuple(k for k in keys if _has_data(df, _COLS[k]))
            else:
                chosen = tuple(k for k in keys
                               if k in self.metrics and _has_data(df, _COLS[k]))
            if chosen:
                blocks.append((chosen, panel))
        return blocks

    def _format_xaxis(self, df, axes, tcol, t_end) -> None:
        t0 = df[tcol].min()
        t1 = df[t_end].max() if t_end in df else df[tcol].max()
        dur = max(t1 - t0, 1e-9)
        step = 1 if dur <= 10 else 10
        ticks = np.arange(math.floor(t0), math.ceil(t1) + step, step)
        for ax in axes:
            ax.set_xticks(ticks)
            ax.set_xlim(t0 - 0.01 * dur, t1 + 0.01 * dur)
            ax.set_xlabel("Time (s)", color="black")
            ax.tick_params(axis="x", colors="black")
            ax.grid(True, axis="x")

    def _resolve_prefix(self, prefix: Optional[str], out_dir: Optional[str]) -> str:
        if prefix is None:
            base = Path(self.result.source or "iperf").stem
            prefix = f"{base}.plots"
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            prefix = os.path.join(out_dir, os.path.basename(prefix))
        return prefix

    def _title_text(self) -> str:
        m = self.result.metadata
        fsz, funit = auto_scale(float(m.get("test_start.bytes", 0) or 0), BYTES)
        snd_val, snd_unit = auto_scale(float(m.get("sndbuf_actual", 0) or 0), BYTES)
        rcv_val, rcv_unit = auto_scale(float(m.get("rcvbuf_actual", 0) or 0), BYTES)
        sender = "server" if int(m.get("test_start.reverse", 0) or 0) == 1 else "client"
        line1 = (f"Time: {m.get('timestamp', '')}    "
                 f"File: {fsz:.2f}{funit}    version: {m.get('version', '')}")
        line2 = (f"Server: {m.get('remote_host', '')}:{int(m.get('remote_port', 0) or 0)}    "
                 f"Client: {m.get('local_host', '')}    Sender: {sender}")
        line3 = (f"Protocol: {m.get('test_start.protocol', '')}    "
                 f"SendBuf: {snd_val:.2f}{snd_unit}    RecvBuf: {rcv_val:.2f}{rcv_unit}    "
                 f"MSS: {int(m.get('tcp_mss_default', 0) or 0)}")
        return "\n".join((line1, line2, line3))

    # -- panels ----------------------------------------------------------- #

    def _plot_data_rate(self, df: pd.DataFrame, ax: Axes, keys) -> None:
        raw_max = max(df[_COLS[k]].max() for k in keys if _has_data(df, _COLS[k]))
        _, unit = auto_scale(raw_max, _DATA_RATE_UNITS)
        scale = next(t for t, u in _DATA_RATE_UNITS if u == unit)
        for k, color in zip(keys, ("tab:blue", "tab:orange")):
            if _has_data(df, _COLS[k]):
                ax.plot(df[_COLS["t"]], df[_COLS[k]] / scale,
                        linewidth=1.5, color=color, label=_LABELS[k])
        ax.set_ylabel(f"Data rate ({unit})")
        ax.legend(loc="upper left", frameon=True, framealpha=0.8)

    def _plot_cwnd(self, df: pd.DataFrame, ax_left: Axes, keys) -> None:
        raw_max = max(df[_COLS[k]].max() for k in keys if _has_data(df, _COLS[k]))
        scale, unit = auto_scale(raw_max, BYTES)
        # auto_scale returns the scaled value; recover the divisor by unit.
        scale = next(t for t, u in BYTES if u == unit)
        handles, labels = [], []
        if "cwnd" in keys and _has_data(df, _COLS["cwnd"]):
            line, = ax_left.plot(df[_COLS["t"]], df[_COLS["cwnd"]] / scale,
                                 linewidth=1.5, color="tab:green")
            handles.append(line)
            labels.append(_LABELS["cwnd"])
            ax_left.set_ylabel(f"CWND ({unit})", color="tab:green")
            ax_left.tick_params(labelcolor="tab:green")
        if "swnd" in keys and _has_data(df, _COLS["swnd"]):
            ax2 = ax_left.twinx() if "cwnd" in keys else ax_left
            line2, = ax2.plot(df[_COLS["t"]], df[_COLS["swnd"]] / scale,
                              linewidth=1.5, color="tab:red")
            handles.append(line2)
            labels.append(_LABELS["swnd"])
            ax2.set_ylabel(f"Send Window ({unit})", color="tab:red")
            ax2.tick_params(labelcolor="tab:red")
            if ax2 is not ax_left:
                ax2.grid(False)
        ax_left.legend(handles, labels, loc="upper left", frameon=True, framealpha=0.8)

    def _plot_rtt(self, df: pd.DataFrame, ax: Axes, keys) -> None:
        handles, labels = [], []
        has_rtt = "rtt" in keys and _has_data(df, _COLS["rtt"])
        has_retx = "retx" in keys and _has_data(df, _COLS["retx"])
        if has_retx:
            ax2 = ax.twinx() if has_rtt else ax
            ax2.bar(df[_COLS["t"]], df[_COLS["retx"]], width=df["interval"],
                    align="edge", alpha=0.3, color="tab:grey", zorder=1)
            handles.append(Patch(facecolor="tab:grey", alpha=0.3))
            labels.append(_LABELS["retx"])
            ax2.set_ylabel("Retransmissions", color="tab:grey")
            ax2.tick_params(labelcolor="tab:grey")
            if ax2 is not ax:
                ax2.grid(False)
        if has_rtt:
            if "rtt_var" in keys and _has_data(df, _COLS["rtt_var"]):
                lower = (df["rtt_ms"] - df["rtt_var_ms"]).clip(lower=0)
                upper = df["rtt_ms"] + df["rtt_var_ms"]
                band = ax.fill_between(df["mid_t"], lower, upper, alpha=0.25,
                                       color="tab:purple", label=_LABELS["rtt_var"],
                                       zorder=2)
                handles.append(band)
                labels.append(_LABELS["rtt_var"])
            line, = ax.plot(df["mid_t"], df["rtt_ms"], linewidth=1.5,
                            color="tab:purple", zorder=3)
            handles.append(line)
            labels.append(_LABELS["rtt"])
            ax.set_ylabel("RTT (ms)", color="tab:purple")
            ax.tick_params(labelcolor="tab:purple")
        ax.legend(handles, labels, loc="upper left", frameon=True, framealpha=0.8)
