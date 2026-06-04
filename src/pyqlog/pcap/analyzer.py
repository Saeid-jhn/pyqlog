"""pcap analyzer for UDP/TCP/QUIC traffic, backed by TShark.

Runs ``tshark`` once to extract a tidy packet table, then derives a per-port
throughput table (plus total and TCP-error counts) and an optional TCP-sequence
table. Port *filtering* is deferred to the plotter so a single analysis can be
re-plotted for different port subsets without re-reading the capture.
"""

from __future__ import annotations

import io
import logging
import shutil
import subprocess
from typing import List, Optional

import pandas as pd

from ..core import AnalysisResult

logger = logging.getLogger(__name__)

TSHARK = shutil.which("tshark") or "/usr/bin/tshark"

FIELDS: List[str] = [
    "frame.time_epoch",
    "frame.len",
    "ip.proto",
    "ip.src",
    "ip.dst",
    "tcp.srcport",
    "tcp.dstport",
    "udp.srcport",
    "udp.dstport",
    "tcp.seq",
    "tcp.analysis.retransmission",
    "tcp.analysis.out_of_order",
    "tcp.analysis.fast_retransmission",
]

_ERROR_FLAGS = [
    "tcp.analysis.retransmission",
    "tcp.analysis.out_of_order",
    "tcp.analysis.fast_retransmission",
]


def tshark_available() -> bool:
    return shutil.which("tshark") is not None


def pcap_to_df(pcap: str, display_filter: Optional[str] = None) -> pd.DataFrame:
    """Run TShark once and return a tidy DataFrame of packet fields."""
    cmd = [TSHARK, "-r", pcap, "-n", "-T", "fields",
           "-E", "header=y", "-E", "separator=,", "-E", "occurrence=f"]
    if display_filter:
        cmd += ["-Y", display_filter]
    for f in FIELDS:
        cmd += ["-e", f]

    logger.debug("Running: %s", " ".join(cmd))
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    df = pd.read_csv(io.StringIO(out))

    df.rename(columns={
        "frame.time_epoch": "Timestamp",
        "frame.len": "Length",
        "ip.src": "SrcIP",
        "ip.dst": "DstIP",
    }, inplace=True)
    if not df.empty:
        df["Timestamp"] -= df["Timestamp"].iloc[0]

    df["SrcPort"] = df["tcp.srcport"].fillna(df["udp.srcport"]).astype("Int64")
    df["DstPort"] = df["tcp.dstport"].fillna(df["udp.dstport"]).astype("Int64")
    df["Protocol"] = df["ip.proto"].map({6: "TCP", 17: "UDP"}).fillna("Other")
    return df


class PcapAnalyzer:
    """Analyze a pcap into throughput + sequence tables."""

    def __init__(self, pcap: str, *, interval: float = 1.0,
                 stream_index: Optional[int] = None,
                 tcp_only: bool = False, quic_only: bool = False):
        self.pcap = pcap
        self.interval = interval
        self.stream_index = stream_index
        self.tcp_only = tcp_only
        self.quic_only = quic_only

    def analyze(self) -> AnalysisResult:
        display_filter = "tcp" if self.tcp_only else "quic" if self.quic_only else None
        df = pcap_to_df(self.pcap, display_filter)

        if self.stream_index is not None and "tcp.stream" in df.columns:
            df = df[df["tcp.stream"] == self.stream_index]

        throughput = self._throughput(df)
        sequence = self._sequence(df)

        return AnalysisResult(
            kind="pcap",
            tables={"throughput": throughput, "sequence": sequence},
            metadata={"interval": self.interval},
            source=self.pcap,
        )

    def _throughput(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = ["Time_Bin", "Port", "Throughput", "TotalThroughput", "Errors"]
        if df.empty:
            return pd.DataFrame(columns=cols)

        work = df.copy()
        work["Time_Bin"] = (work["Timestamp"] // self.interval) * self.interval

        total = (work.groupby("Time_Bin")["Length"].sum()
                 .mul(8 / self.interval).rename("TotalThroughput").reset_index())

        port = (work[["Time_Bin", "SrcPort", "Length"]]
                .rename(columns={"SrcPort": "Port"})
                .groupby(["Time_Bin", "Port"])["Length"].sum()
                .mul(8 / self.interval).rename("Throughput").reset_index())

        result = port.merge(total, on="Time_Bin", how="left")

        work["ErrorEvent"] = work[_ERROR_FLAGS].notna().any(axis=1).astype(int)
        errors = (work.groupby("Time_Bin")["ErrorEvent"].sum()
                  .rename("Errors").reset_index())
        result = result.merge(errors, on="Time_Bin", how="left")
        return result

    @staticmethod
    def _sequence(df: pd.DataFrame) -> pd.DataFrame:
        cols = ["Timestamp", "SequenceNumber", "SrcIP", "DstIP", "SrcPort", "DstPort"]
        if df.empty or "tcp.seq" not in df.columns:
            return pd.DataFrame(columns=cols)
        seq = df[df["tcp.seq"].notna()]
        if seq.empty:
            return pd.DataFrame(columns=cols)
        return (seq[["Timestamp", "tcp.seq", "SrcIP", "DstIP", "SrcPort", "DstPort"]]
                .rename(columns={"tcp.seq": "SequenceNumber"})
                .reset_index(drop=True))
