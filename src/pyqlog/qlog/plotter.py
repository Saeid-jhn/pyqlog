"""Render qlog metrics from an :class:`AnalysisResult` (no CSV round-trip required).

Produces a single stacked figure with up to five panels, in this top-to-bottom
visual order: offset/retransmissions/cumulative bytes, throughput & goodput,
pacing rate, CWND & bytes-in-flight, and RTTs. Panels with no data are skipped.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable, List, Optional

import matplotlib.pyplot as plt
import pandas as pd

from ..core import AnalysisResult, apply_style, save_figure

logger = logging.getLogger(__name__)

MB = 1e6


class QlogPlotter:
    """Plot a qlog :class:`AnalysisResult` to one or more image formats."""

    def __init__(self, result: AnalysisResult):
        if result.kind != "qlog":
            raise ValueError(f"QlogPlotter needs a qlog result, got {result.kind!r}")
        self.result = result

    def render(
        self,
        *,
        out_dir: Optional[str] = None,
        formats: Iterable[str] = ("png",),
        prefix: Optional[str] = None,
    ) -> List[Path]:
        apply_style()

        df_packets = self.result.table("packets").copy()
        df_metrics = self.result.table("metrics").copy()
        df_offsets = self.result.table("offsets").copy()
        df_rate = self.result.table("data_rate").copy()

        # Derived seconds / MB columns.
        if "time" in df_offsets:
            df_offsets["time_s"] = df_offsets["time"] / 1e6
            df_offsets["offset_MB"] = df_offsets["offset"] / 1e6
        if "time" in df_packets:
            df_packets["time_s"] = df_packets["time"] / 1e6
            if "packet_size_cumsum" not in df_packets.columns:
                df_packets["packet_size_cumsum"] = df_packets["packet_size"].cumsum()
            df_packets["packet_size_cumsum_MB"] = df_packets["packet_size_cumsum"] / 1e6
        if "time" in df_metrics:
            df_metrics["time_s"] = df_metrics["time"] / 1e6

        # Decide which panels have data, in visual order.
        panels = []
        if not df_offsets.empty or not df_packets.empty:
            panels.append(self._panel_offset)
        if not df_rate.empty:
            panels.append(self._panel_data_rate)
        if self._has_metric(df_metrics, "pacing_rate"):
            panels.append(self._panel_pacing)
        if self._has_metric(df_metrics, "cwnd", "congestion_window", "bytes_in_flight"):
            panels.append(self._panel_cwnd)
        if self._has_metric(df_metrics, "smoothed_rtt", "latest_rtt", "min_rtt"):
            panels.append(self._panel_rtt)

        if not panels:
            logger.info("Nothing to plot for %s", self.result.source)
            return []

        fig, axes = plt.subplots(len(panels), 1, sharex=False,
                                 figsize=(5, max(3, 2.4 * len(panels))),
                                 squeeze=False)
        axes = axes.flatten()
        for ax, draw in zip(axes, panels):
            ax.grid(True)
            draw(ax, df_packets, df_metrics, df_offsets, df_rate)

        fig.align_ylabels(axes)
        fig.tight_layout()

        prefix = self._resolve_prefix(prefix, out_dir)
        return save_figure(fig, prefix, formats)

    # -- helpers ---------------------------------------------------------- #

    @staticmethod
    def _has_metric(df: pd.DataFrame, *keys: str) -> bool:
        return "key" in df.columns and df["key"].isin(keys).any()

    def _resolve_prefix(self, prefix: Optional[str], out_dir: Optional[str]) -> str:
        if prefix is None:
            base = os.path.basename(self.result.source or "qlog")
            prefix = base
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            prefix = os.path.join(out_dir, os.path.basename(prefix))
        return prefix

    # -- panels ----------------------------------------------------------- #

    def _panel_offset(self, ax, df_packets, df_metrics, df_offsets, df_rate):
        handles = []
        if not df_offsets.empty and "duplicate" in df_offsets:
            off = df_offsets[~df_offsets["duplicate"]]
            retx = df_offsets[df_offsets["duplicate"]]
            handles.append(ax.plot(off["time_s"], off["offset_MB"], ".",
                                   markersize=1, label="Offset")[0])
            handles.append(ax.plot(retx["time_s"], retx["offset_MB"], ".",
                                   markersize=1, label="Retransmitted Offset")[0])
        if not df_packets.empty:
            handles.append(ax.plot(df_packets["time_s"],
                                   df_packets["packet_size_cumsum_MB"], ".",
                                   markersize=1, label="Cumulative Data Size")[0])
        if handles:
            ax.legend(handles=handles, markerscale=8)
        ax.set_ylabel("offset [MB]")
        ax.set_xlabel("Time [s]")

    def _panel_data_rate(self, ax, df_packets, df_metrics, df_offsets, df_rate):
        tput = ax.plot(df_rate["start_interval (s)"],
                       df_rate["throughput (bps)"] / MB, "-",
                       label="Throughput")[0]
        gput = ax.plot(df_rate["start_interval (s)"],
                       df_rate["goodput (bps)"] / MB, "--",
                       label="Goodput")[0]
        ax.legend(handles=[tput, gput])
        ax.set_ylabel("data rate [Mbps]")
        ax.set_xlabel("Time [s]")

    def _panel_pacing(self, ax, df_packets, df_metrics, df_offsets, df_rate):
        pacing = df_metrics[df_metrics["key"] == "pacing_rate"]
        line = ax.plot(pacing["time_s"], pacing["value"] / MB, ".",
                       markersize=1, label="Pacing Rate")[0]
        ax.legend(handles=[line], markerscale=8)
        ax.set_ylabel("pacing rate [Mbps]")
        ax.set_xlabel("Time [s]")

    def _panel_cwnd(self, ax, df_packets, df_metrics, df_offsets, df_rate):
        cwnd = df_metrics[df_metrics["key"].isin(["cwnd", "congestion_window"])]
        flight = df_metrics[df_metrics["key"] == "bytes_in_flight"]
        handles = []
        if not cwnd.empty:
            handles.append(ax.plot(cwnd["time_s"], cwnd["value"] / MB, ".",
                                   markersize=1, label="CWND")[0])
        if not flight.empty:
            handles.append(ax.plot(flight["time_s"], flight["value"] / MB, ".",
                                   markersize=1, label="Bytes in Flight")[0])
        if handles:
            ax.legend(handles=handles, markerscale=8)
        ax.set_ylabel("Metrics [MB]")
        ax.set_xlabel("Time [s]")

    def _panel_rtt(self, ax, df_packets, df_metrics, df_offsets, df_rate):
        series = {
            "smoothed_rtt": "Smoothed RTT",
            "latest_rtt": "Latest RTT",
            "min_rtt": "Min RTT",
        }
        handles = []
        mins, p99s = [], []
        for key, label in series.items():
            sub = df_metrics[df_metrics["key"] == key]
            if sub.empty:
                continue
            handles.append(ax.plot(sub["time_s"], sub["value"] / 1e3, ".",
                                   markersize=1, label=label)[0])
            mins.append(sub["value"].min())
            p99s.append(sub["value"].quantile(0.99))
        if handles:
            ax.legend(handles=handles, markerscale=8)
            # Clip y to the 99th percentile to keep spikes from flattening the plot.
            ax.set_ylim((min(mins) * 0.9) / 1e3, (max(p99s) * 1.2) / 1e3)
        ax.set_ylabel("RTT [ms]")
        ax.set_xlabel("Time [s]")
