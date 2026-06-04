"""Render pcap throughput and TCP-sequence figures from an :class:`AnalysisResult`."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import EngFormatter

from ..core import AnalysisResult, apply_style, save_figure

logger = logging.getLogger(__name__)


class PcapPlotter:
    """Plot a pcap :class:`AnalysisResult`.

    Args:
        result:   the pcap analysis.
        ports:    source ports to draw separately (empty -> all available).
        legends:  optional ``{port: label}`` overrides.
        total:    include the total-throughput line.
        sequence: also render the TCP sequence-number scatter.
        tcp_error: overlay scaled TCP-error bars on the throughput plot.
    """

    def __init__(self, result: AnalysisResult, *,
                 ports: Optional[List[int]] = None,
                 legends: Optional[Dict[int, str]] = None,
                 total: bool = False, sequence: bool = False,
                 tcp_error: bool = False):
        if result.kind != "pcap":
            raise ValueError(f"PcapPlotter needs a pcap result, got {result.kind!r}")
        self.result = result
        self.ports = ports or []
        self.legends = legends or {}
        self.total = total
        self.sequence = sequence
        self.tcp_error = tcp_error
        self.interval = float(result.metadata.get("interval", 1.0))

    def render(self, *, out_dir: Optional[str] = None,
               formats: Iterable[str] = ("png",),
               prefix: Optional[str] = None) -> List[Path]:
        apply_style()
        formats = list(formats)
        prefix = self._resolve_prefix(prefix, out_dir)

        written: List[Path] = []
        written += self._render_throughput(self.result.table("throughput"),
                                           prefix, formats)
        if self.sequence:
            written += self._render_sequence(self.result.table("sequence"),
                                             prefix, formats)
        return written

    # -- panels ----------------------------------------------------------- #

    def _ports_to_plot(self, df: pd.DataFrame) -> List[int]:
        if self.ports:
            return self.ports
        return sorted(p for p in df["Port"].dropna().unique())

    def _render_throughput(self, df: pd.DataFrame, prefix: str,
                           formats: List[str]) -> List[Path]:
        if df.empty:
            logger.info("No throughput data for %s", self.result.source)
            return []

        fig, ax = plt.subplots(figsize=(12, 6))
        for port in self._ports_to_plot(df):
            sub = df[df["Port"] == port]
            if not sub.empty:
                ax.plot(sub["Time_Bin"], sub["Throughput"],
                        label=self.legends.get(port, f"Port {port}"))
        if self.total and "TotalThroughput" in df.columns:
            tot = df[["Time_Bin", "TotalThroughput"]].drop_duplicates()
            ax.plot(tot["Time_Bin"], tot["TotalThroughput"], "--", label="Total")

        ax.set_ylabel("Throughput (bits/s)")
        ax.yaxis.set_major_formatter(EngFormatter(unit="b"))
        ax.set_xlabel("Time (s)")

        if self.tcp_error and "Errors" in df.columns and df["Errors"].sum() > 0:
            self._overlay_errors(ax, df)

        ax.legend(loc="upper right")
        ax.grid(True)
        return save_figure(fig, f"{prefix}.data_rate", formats)

    def _overlay_errors(self, ax, df: pd.DataFrame) -> None:
        max_tp = (df["TotalThroughput"].max() if "TotalThroughput" in df.columns
                  else df["Throughput"].max())
        max_err = df["Errors"].max()
        scale = (max_tp / max_err) * 0.2 if max_err and max_tp else 1
        err = df[["Time_Bin", "Errors"]].drop_duplicates()
        ax.bar(err["Time_Bin"], err["Errors"] * scale, width=self.interval * 0.8,
               alpha=0.6, color="red", label="TCP Errors")

    def _render_sequence(self, df: pd.DataFrame, prefix: str,
                         formats: List[str]) -> List[Path]:
        if df.empty:
            return []
        fig, ax = plt.subplots(figsize=(12, 6))
        if self.ports:
            for port in self.ports:
                sub = df[df["SrcPort"] == port]
                if not sub.empty:
                    ax.scatter(sub["Timestamp"], sub["SequenceNumber"], s=5,
                               label=self.legends.get(port, f"Port {port}"))
        else:
            ax.scatter(df["Timestamp"], df["SequenceNumber"], s=5,
                       label="TCP Sequence")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Sequence Number")
        ax.legend()
        ax.grid(True)
        return save_figure(fig, f"{prefix}.seq", formats)

    def _resolve_prefix(self, prefix: Optional[str], out_dir: Optional[str]) -> str:
        if prefix is None:
            prefix = os.path.basename(self.result.source or "capture")
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            prefix = os.path.join(out_dir, os.path.basename(prefix))
        return prefix
