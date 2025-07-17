#!/usr/bin/env python3
"""
pcap_inspector.py
-----------------

Comprehensive pcap inspector for UDP, TCP, and QUIC:
- Throughput plots per source port and total
- Optional TCP-sequence scatter
- Optional TCP-error bar chart (scaled, in red)
- Flexible output formats: png, csv, svg, pdf
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import shutil
import subprocess
import time
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.ticker import FuncFormatter

# Increase default font sizes for publication-quality
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12
})
# Set up Seaborn and layout
sns.set()
sns.set_theme(style='whitegrid')

# -------------------------------------------------------------------------- #
# Config                                                                      #
# -------------------------------------------------------------------------- #

TSHARK = shutil.which("tshark") or "/usr/bin/tshark"

FIELDS = [
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
    # TCP analysis flags for error plotting
    "tcp.analysis.retransmission",
    "tcp.analysis.out_of_order",
    "tcp.analysis.fast_retransmission",
]

# -------------------------------------------------------------------------- #
# TShark CSV reader                                                           #
# -------------------------------------------------------------------------- #


def pcap_to_df(pcap: str, display_filter: str | None = None) -> pd.DataFrame:
    """Run TShark once and return a tidy DataFrame."""
    cmd = [
        TSHARK,
        "-r", pcap,
        "-n",
        "-T", "fields",
        "-E", "header=y",
        "-E", "separator=,",
        "-E", "occurrence=f",
    ]
    if display_filter:
        cmd += ["-Y", display_filter]
    for f in FIELDS:
        cmd += ["-e", f]

    logging.debug("Running: %s", " ".join(cmd))
    out = subprocess.run(cmd, capture_output=True,
                         text=True, check=True).stdout
    df = pd.read_csv(io.StringIO(out))

    df.rename(
        columns={
            "frame.time_epoch": "Timestamp",
            "frame.len": "Length",
            "ip.src": "SrcIP",
            "ip.dst": "DstIP",
        },
        inplace=True,
    )
    df["Timestamp"] -= df["Timestamp"].iloc[0]

    # Generic port columns usable for UDP & TCP
    df["SrcPort"] = df["tcp.srcport"].fillna(df["udp.srcport"]).astype("Int64")
    df["DstPort"] = df["tcp.dstport"].fillna(df["udp.dstport"]).astype("Int64")

    df["Protocol"] = df["ip.proto"].map({6: "TCP", 17: "UDP"}).fillna("Other")

    # Simple QUIC heuristic
    quic_mask = (df["Protocol"] == "UDP") & (
        df[["SrcPort", "DstPort"]].isin(
            {443, 784, 4433, 50001, 50002}).any(axis=1)
    )
    df.loc[quic_mask, "Protocol"] = "QUIC"
    return df

# -------------------------------------------------------------------------- #
# Classes                                                                     #
# -------------------------------------------------------------------------- #


class PcapFileProcessor:
    def __init__(
        self,
        pcap: str,
        stream_index: Optional[int],
        tcp_only: bool,
        quic_only: bool,
    ):
        self.pcap = pcap
        self.stream_index = stream_index
        self.tcp_only = tcp_only
        self.quic_only = quic_only

    def read(self,) -> Tuple[
        List[Tuple[float, int, str, int, int,
                   Optional[bool], Optional[bool], Optional[bool]]],
        List[Tuple[float, int, str, str, int, int]],
    ]:
        if self.tcp_only:
            df = pcap_to_df(self.pcap, "tcp")
        elif self.quic_only:
            df = pcap_to_df(self.pcap, "quic")
        else:
            df = pcap_to_df(self.pcap)

        if self.stream_index is not None and "tcp.stream" in df.columns:
            df = df[df["tcp.stream"] == self.stream_index]

        packets = df[
            ["Timestamp", "Length", "Protocol", "SrcPort", "DstPort",
             "tcp.analysis.retransmission", "tcp.analysis.out_of_order", "tcp.analysis.fast_retransmission"]
        ].itertuples(index=False, name=None)

        seq_df = df[df["tcp.seq"].notna()]
        seq = seq_df[
            ["Timestamp", "tcp.seq", "SrcIP", "DstIP", "SrcPort", "DstPort"]
        ].itertuples(index=False, name=None)
        return list(packets), list(seq)


class ThroughputCalculator:
    def __init__(self, interval: float, ports: Optional[List[int]], tcp_error: bool):
        self.interval = interval
        self.ports = ports
        self.tcp_error = tcp_error

    def calculate(self, packets) -> pd.DataFrame:
        cols_full = [
            "Timestamp", "Length", "Protocol", "SrcPort", "DstPort",
            "tcp.analysis.retransmission", "tcp.analysis.out_of_order", "tcp.analysis.fast_retransmission"
        ]
        df = pd.DataFrame(packets, columns=cols_full)

        if not self.tcp_error:
            df = df[["Timestamp", "Length", "Protocol", "SrcPort", "DstPort"]]

        df["Time_Bin"] = (df["Timestamp"] // self.interval) * self.interval

        total = (
            df.groupby("Time_Bin")["Length"]
            .sum()
            .mul(8 / self.interval)
            .rename("TotalThroughput")
            .reset_index()
        )

        port_df = df[["Time_Bin", "SrcPort", "Length"]].rename(
            columns={"SrcPort": "Port"})
        if self.ports:
            port_df = port_df[port_df["Port"].isin(self.ports)]
        port = (
            port_df.groupby(["Time_Bin", "Port"])["Length"]
                   .sum()
                   .mul(8 / self.interval)
                   .rename("Throughput")
                   .reset_index()
        )

        result = port.merge(total, on="Time_Bin", how="left")

        if self.tcp_error:
            df["ErrorEvent"] = df[[
                "tcp.analysis.retransmission",
                "tcp.analysis.out_of_order",
                "tcp.analysis.fast_retransmission"
            ]].notna().any(axis=1).astype(int)
            errors = (
                df.groupby("Time_Bin")["ErrorEvent"]
                  .sum()
                  .rename("Errors")
                  .reset_index()
            )
            result = result.merge(errors, on="Time_Bin", how="left")

        return result


class Plotter:
    def __init__(self, base_name: str, plot_total: bool, legends: Dict[int, str], interval: float, formats: List[str]):
        self.base_name = f"{base_name}.pcap"
        self.plot_total = plot_total
        self.legends = legends
        self.interval = interval
        self.formats = formats

    def throughput(self, df: pd.DataFrame, ports: List[int]) -> None:
        if df.empty:
            logging.info("No throughput data for %s", self.base_name)
            return

        fig, ax1 = plt.subplots(figsize=(12, 6))
        for port in ports:
            sub = df[df["Port"] == port]
            if not sub.empty:
                ax1.plot(sub["Time_Bin"], sub["Throughput"],
                         label=self.legends.get(port, f"Port {port}"))
        if self.plot_total:
            tot = df[["Time_Bin", "TotalThroughput"]].drop_duplicates()
            ax1.plot(tot["Time_Bin"], tot["TotalThroughput"],
                     "--", label="Total")

        ax1.set_ylabel("Throughput (Mbps)")
        ax1.set_xlabel("Time (s)")

        if "Errors" in df.columns and df["Errors"].sum() > 0:
            max_tp = df["TotalThroughput"].max(
            ) if "TotalThroughput" in df.columns else df["Throughput"].max()
            max_err = df["Errors"].max()
            scale = (max_tp / max_err) * 0.2 if max_err and max_tp else 1
            err = df[["Time_Bin", "Errors"]].drop_duplicates()
            heights = err["Errors"] * scale
            ax1.bar(err["Time_Bin"], heights, width=self.interval*0.8,
                    alpha=0.6, color='red', label="TCP Errors")

        ax1.legend(loc="upper right")
        ax1.grid(True)

        # save outputs per requested formats
        if "png" in self.formats:
            fig.savefig(f"{self.base_name}.data_rate.png", dpi=900)
        if "svg" in self.formats:
            fig.savefig(f"{self.base_name}.data_rate.svg")
        if "pdf" in self.formats:
            fig.savefig(f"{self.base_name}.data_rate.pdf")
        plt.close(fig)

    def sequence(self, seq: List[Tuple[float, int, str, str, int, int]], ports: List[int]) -> None:
        if not seq:
            return
        df = pd.DataFrame(seq, columns=["Timestamp", "SequenceNumber", "SrcIP", "DstIP", "SrcPort", "DstPort"])\

        plt.figure(figsize=(12, 6))
        if ports:
            for port in ports:
                sub = df[df["SrcPort"] == port]
                if not sub.empty:
                    plt.scatter(sub["Timestamp"], sub["SequenceNumber"],
                                s=5, label=self.legends.get(port, f"Port {port}"))
        else:
            plt.scatter(df["Timestamp"], df["SequenceNumber"],
                        s=5, label="TCP Sequence")

        plt.xlabel("Time (s)", fontsize=14)
        plt.ylabel("Sequence Number", fontsize=14)
        plt.tick_params(axis='both', labelsize=12)
        plt.legend()
        plt.grid(True)

        # save sequence outputs
        if "png" in self.formats:
            plt.savefig(f"{self.base_name}.seq.png")
        if "svg" in self.formats:
            plt.savefig(f"{self.base_name}.seq.svg")
        if "pdf" in self.formats:
            plt.savefig(f"{self.base_name}.seq.pdf")
        plt.close()


class PcapAnalyzer:
    def __init__(self, pcap: str, interval: float, stream_index: Optional[int], tcp_only: bool, quic_only: bool,
                 ports: Optional[List[int]], total: bool, legends: Dict[int, str], out_dir: Optional[str],
                 do_seq: bool, tcp_error: bool, formats: List[str]):
        if out_dir is None:
            out_dir = os.path.dirname(pcap)
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.join(out_dir, os.path.splitext(
            os.path.basename(pcap))[0])
        self.processor = PcapFileProcessor(
            pcap, stream_index, tcp_only, quic_only)
        self.calc = ThroughputCalculator(interval, ports, tcp_error)
        self.plot = Plotter(base, total, legends, interval, formats)
        self.ports = ports or []
        self.do_seq = do_seq
        self.tcp_error = tcp_error
        self.formats = formats

    def run(self) -> None:
        t0 = time.time()
        logging.info("▶ analysing %s", self.processor.pcap)

        packets, seq = self.processor.read()
        tp_df = self.calc.calculate(packets)
        # write CSV only if requested
        if "csv" in self.formats:
            tp_df.to_csv(f"{self.plot.base_name}.data_rate.csv", index=False)

        self.plot.throughput(tp_df, self.ports)

        if self.do_seq and seq:
            if "csv" in self.formats:
                pd.DataFrame(seq, columns=["Timestamp", "SequenceNumber", "SrcIP", "DstIP", "SrcPort", "DstPort"]).to_csv(
                    f"{self.plot.base_name}.seq.csv", index=False)
            self.plot.sequence(seq, self.ports)

        logging.info("✓ done in %.2f s", time.time() - t0)

# -------------------------------------------------------------------------- #
# CLI                                                                         #
# -------------------------------------------------------------------------- #


def wrapper(pcap, interval, stream, tcp, quic, ports, total, legends, out_dir, do_seq, tcp_error, formats):
    if not os.path.exists(pcap):
        logging.error("File not found: %s", pcap)
        return
    PcapAnalyzer(pcap, interval, stream, tcp, quic, ports, total,
                 legends, out_dir, do_seq, tcp_error, formats).run()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Comprehensive pcap inspector")
    parser.add_argument("pcap_files", nargs='+', help="pcap file(s)")
    parser.add_argument("--interval", type=float,
                        default=1.0, help="bin width (s)")
    parser.add_argument("--stream-index", type=int, help="filter a TCP stream")
    parser.add_argument("--tcp", action="store_true", help="analyse only TCP")
    parser.add_argument("--quic", action="store_true",
                        help="analyse only QUIC")
    parser.add_argument("--port", type=int, nargs='+',
                        help="ports to plot separately")
    parser.add_argument("--port-legend", type=str, nargs='+',
                        help="custom legend text per port")
    parser.add_argument("--total", action="store_true",
                        help="show total throughput line")
    parser.add_argument("--sequence", action="store_true",
                        help="draw TCP seq scatter")
    parser.add_argument("--tcp-error", action="store_true",
                        help="extract TCP-analysis flags and plot error counts")
    parser.add_argument(
        "--formats",
        nargs='+',
        choices=['png', 'csv', 'svg', 'pdf'],
        default=['png'],
        help="output formats to generate (default: png only)",
    )
    parser.add_argument("--output-dir", default=None,
                        help="folder for output (default: beside pcap)")
    args = parser.parse_args()

    legends = (
        dict(zip(args.port, args.port_legend))
        if args.port_legend else {p: f"Port {p}" for p in args.port or []}
    )

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    with Pool(cpu_count()) as pool:
        pool.starmap(
            wrapper,
            [(
                pcap,
                args.interval,
                args.stream_index,
                args.tcp,
                args.quic,
                args.port,
                args.total,
                legends,
                args.output_dir,
                args.sequence,
                args.tcp_error,
                args.formats,
            ) for pcap in args.pcap_files]
        )


if __name__ == "__main__":
    main()
