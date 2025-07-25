from __future__ import annotations
import argparse
import csv
import json
import logging
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple

#!/usr/bin/env python3
"""
iperf_log_processor.py
───────────────────────────────────────────────────────────────────────────────
Convert iperf3 **JSON‑L** (streaming) logs to a tidy CSV – and optionally plot.

 • Handles plain sender logs **or** client logs captured with
   `iperf3 --get-server-output`, i.e. a single file containing *both*
   sender‑ and receiver‑side interval records.
 • Produces one row per (start,end) interval; both throughput columns are kept:
       rcv_goodput (bps)      – only on receiver intervals
       snd_throughput (bps)   – on sender intervals (including embedded server intervals)
 • Adds sender TCP metrics when present: retransmits, snd_cwnd, snd_wnd, RTT …

CSV column order
────────────────
start_time (s), end_time (s), rcv_goodput (bps), snd_throughput (bps),
retransmits, snd_cwnd (K), snd_wnd (K), rtt (us), rtt_var (us)

Usage
─────
    python iperf_log_processor.py [-v|-vv] [--plot] <log1> [log2 ...]
"""


@dataclass(slots=True)
class IntervalRow:
    """One row in the output CSV, keyed by (start,end)."""
    start_time_sec: float
    end_time_sec: float

    # throughput columns
    rcv_goodput_bps:    Optional[int] = None
    snd_throughput_bps: Optional[int] = None

    # TCP metrics (sender only)
    retransmits:  Optional[int] = None
    snd_cwnd_k:   Optional[float] = None
    snd_wnd_k:    Optional[float] = None
    rtt_us:       Optional[int] = None
    rtt_var_us:   Optional[int] = None

    # column ↔︎ attribute map for CSV export
    _MAP: ClassVar[Dict[str, str]] = {
        "start_time (s)":       "start_time_sec",
        "end_time (s)":         "end_time_sec",
        "rcv_goodput (bps)":    "rcv_goodput_bps",
        "snd_throughput (bps)": "snd_throughput_bps",
        "retransmits":          "retransmits",
        "snd_cwnd (K)":         "snd_cwnd_k",
        "snd_wnd (K)":          "snd_wnd_k",
        "rtt (us)":             "rtt_us",
        "rtt_var (us)":         "rtt_var_us",
    }

    def to_csv(self, cols: Sequence[str]) -> Dict[str, Any]:
        """Serialize only non-None fields into CSV columns."""
        raw = asdict(self)
        return {c: raw[self._MAP[c]] for c in cols if raw[self._MAP[c]] is not None}


class IperfLogProcessor:
    """Convert one iperf3 JSON‑stream log to CSV (+ optional plot)."""

    CSV_COLS: ClassVar[List[str]] = [
        "start_time (s)",
        "end_time (s)",
        "rcv_goodput (bps)",
        "snd_throughput (bps)",
        "retransmits",
        "snd_cwnd (K)",
        "snd_wnd (K)",
        "rtt (us)",
        "rtt_var (us)",
    ]

    def __init__(self, path: Path, plot: bool = False) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"Log file not found: {path}")
        self.path = path
        self.plot = plot
        self.log = logging.getLogger(path.name)

    def run(self) -> Path:
        """Process the log file: parse, write CSV, and optionally plot."""
        rows = self._parse_log()
        rows = self._drop_subsecond_tail(rows)
        csv_path = self._write_csv(rows)
        if self.plot:
            self._plot(csv_path)
        return csv_path

    def _parse_log(self) -> List[IntervalRow]:
        """Merge sender & receiver records (and embedded server intervals) by (start,end)."""
        rows_by_interval: Dict[Tuple[float, float], IntervalRow] = {}

        with self.path.open(encoding="utf-8") as fh:
            for ln, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    self.log.debug("line %d: invalid JSON", ln)
                    continue

                # Determine interval blocks: client or embedded server
                if rec.get("event") == "interval":
                    blocks = [rec["data"]]
                elif rec.get("event") == "server_output_json":
                    blocks = rec["data"].get("intervals", [])
                else:
                    continue

                for blk in blocks:
                    streams = blk.get("streams", [])
                    if not streams:
                        self.log.debug("line %d: no streams in block", ln)
                        continue
                    st = streams[0]

                    # normalize and key by start/end rounded to milliseconds
                    beg = round(st.get("start", 0.0), 3)
                    end = round(st.get("end",   0.0), 3)
                    key = (beg, end)
                    row = rows_by_interval.setdefault(key,
                                                      IntervalRow(beg, end))

                    is_sender = st.get("sender", True)
                    bits_per_sec = int(st.get("bits_per_second", 0))

                    if is_sender:
                        # fill sender throughput & TCP stats
                        row.snd_throughput_bps = bits_per_sec
                        row.retransmits = st.get(
                            "retransmits") or row.retransmits
                        if "snd_cwnd" in st:
                            row.snd_cwnd_k = st["snd_cwnd"] / 1000
                        if "snd_wnd" in st:
                            row.snd_wnd_k = st["snd_wnd"] / 1000
                        row.rtt_us = st.get("rtt") or row.rtt_us
                        row.rtt_var_us = st.get("rttvar") or row.rtt_var_us
                    else:
                        # receiver goodput
                        row.rcv_goodput_bps = bits_per_sec

        self.log.info("parsed %d interval rows", len(rows_by_interval))
        return sorted(rows_by_interval.values(), key=lambda r: r.start_time_sec)

    def _drop_subsecond_tail(self, rows: List[IntervalRow]) -> List[IntervalRow]:
        """Drop final interval if shorter than 1 second."""
        if rows and (rows[-1].end_time_sec - rows[-1].start_time_sec < 1.0):
            self.log.info(
                "dropping final sub‑second row %s-%s",
                rows[-1].start_time_sec,
                rows[-1].end_time_sec,
            )
            rows.pop()
        return rows

    def _write_csv(self, rows: List[IntervalRow]) -> Path:
        """Write out the merged intervals to CSV."""
        csv_path = self.path.with_suffix(self.path.suffix + ".csv")
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=self.CSV_COLS)
            writer.writeheader()
            for row in rows:
                writer.writerow(row.to_csv(self.CSV_COLS))
        self.log.info("wrote CSV → %s", csv_path)
        return csv_path

    def _plot(self, csv_path: Path) -> None:
        """Plot CSV via external iperf3_plotter if installed."""
        try:
            from iperf3_plotter import plot_csv
            self.log.info("plotting %s", csv_path)
            plot_csv(str(csv_path))
        except ImportError:
            self.log.error("iperf3_plotter not installed; skipping plot")
        except Exception as exc:
            self.log.error("plot failed: %s", exc)


def _setup_logging(verbosity: int) -> None:
    level = logging.WARNING - min(verbosity, 2) * 10
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(
        description="Convert iperf3 JSON‑L logs to CSV."
    )
    ap.add_argument("log_files", nargs="+", help="iperf3 JSON‑stream files")
    ap.add_argument("-v", action="count", default=0,
                    help="-v/-vv for verbosity")
    ap.add_argument("--plot", action="store_true",
                    help="plot CSV after writing")
    args = ap.parse_args(argv)

    _setup_logging(args.v)
    root = logging.getLogger("iperf_processor")

    t0 = time.perf_counter()
    for path_str in args.log_files:
        path = Path(path_str)
        try:
            start = time.perf_counter()
            IperfLogProcessor(path, plot=args.plot).run()
            root.info("processed %s in %.3fs", path,
                      time.perf_counter() - start)
        except Exception as exc:
            root.error("%s: %s", path, exc)

    root.info("finished %d file(s) in %.3fs", len(
        args.log_files), time.perf_counter() - t0)


if __name__ == "__main__":
    main(sys.argv[1:])
