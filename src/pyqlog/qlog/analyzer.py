"""QUIC qlog/sqlog analyzer: parse a log into a replottable :class:`AnalysisResult`.

Produces tables: ``packets``, ``metrics``, ``offsets``, ``datagram``,
``data_rate``. Throughput is summed from datagram (on-wire) byte lengths;
goodput from non-duplicate stream-frame offsets only (retransmissions excluded).
Both are binned over a fixed interval with ``numpy.histogram``.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from ..core import AnalysisResult
from .parsers import get_parser

logger = logging.getLogger(__name__)

DATA_RATE_COLUMNS = [
    "start_interval (s)", "end_interval (s)",
    "throughput (bps)", "goodput (bps)",
]


def compute_data_rate(
    df_datagram: pd.DataFrame,
    df_offsets: pd.DataFrame,
    interval: str = "1000ms",
) -> pd.DataFrame:
    """Bin throughput (datagram bytes) and goodput (unique offset bytes) into bps.

    ``interval`` is any pandas timedelta string (e.g. ``"1000ms"``, ``"0.5s"``).
    Timestamps are assumed to be in microseconds.
    """
    empty = pd.DataFrame(columns=DATA_RATE_COLUMNS)

    times_dg = (df_datagram["time"].values.astype(float) / 1e6
                if "time" in df_datagram else np.array([]))
    bytes_dg = (df_datagram["length"].values.astype(float)
                if "length" in df_datagram else np.array([]))

    if "duplicate" in df_offsets.columns:
        mask = (~df_offsets["duplicate"]).values
        times_off = df_offsets["time"].values.astype(float)[mask] / 1e6
        bytes_off = df_offsets["length"].values.astype(float)[mask]
    else:
        times_off = np.array([])
        bytes_off = np.array([])

    if not (times_dg.size or times_off.size):
        return empty

    start = min(times_dg.min() if times_dg.size else float("inf"),
                times_off.min() if times_off.size else float("inf"))
    end = max(times_dg.max() if times_dg.size else float("-inf"),
              times_off.max() if times_off.size else float("-inf"))

    interval_sec = pd.to_timedelta(interval).total_seconds()
    bins = np.arange(start, end + interval_sec, interval_sec)

    dg_sums = np.histogram(times_dg, bins=bins, weights=bytes_dg)[0]
    off_sums = np.histogram(times_off, bins=bins, weights=bytes_off)[0]

    return pd.DataFrame({
        "start_interval (s)": bins[:-1].astype(int),
        "end_interval (s)": bins[1:].astype(int),
        "throughput (bps)": dg_sums * 8 / interval_sec,
        "goodput (bps)": off_sums * 8 / interval_sec,
    })


class QlogAnalyzer:
    """Parse a qlog/sqlog file and build an :class:`AnalysisResult`."""

    def __init__(self, qlog_file: str, *, interval: str = "1000ms"):
        self.qlog_file = qlog_file
        self.interval = interval

    def analyze(self) -> AnalysisResult:
        parser = get_parser(self.qlog_file)
        df_packets, df_metrics, df_offsets, df_datagram = parser.extract()

        data_rate = compute_data_rate(df_datagram, df_offsets, self.interval)

        return AnalysisResult(
            kind="qlog",
            tables={
                "packets": df_packets,
                "metrics": df_metrics,
                "offsets": df_offsets,
                "datagram": df_datagram,
                "data_rate": data_rate,
            },
            metadata={"interval": self.interval},
            source=self.qlog_file,
        )
