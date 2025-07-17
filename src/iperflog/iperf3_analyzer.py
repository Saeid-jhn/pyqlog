#!/usr/bin/env python3
from __future__ import annotations

"""
iperf_log_processor.py – convert iperf3 JSON‑L logs to CSV and (optionally) plot.

Usage:
    python iperf_log_processor.py [-v|-vv] [--plot] <log1> [log2 ...]
"""

import argparse
import csv
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Sequence


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #

@dataclass(slots=True)
class IntervalRow:
    """One iperf3 *interval* event."""

    start_time_sec: float
    end_time_sec:   float
    goodput_bps:    int
    retransmissions: Optional[int] = None
    cwnd_k:          Optional[float] = None
    rtt_us:          Optional[int] = None
    rttvar_us:       Optional[int] = None

    _MAP: ClassVar[Dict[str, str]] = {
        "start time (sec)":      "start_time_sec",
        "end time   (sec)":      "end_time_sec",
        "goodput (bits/sec)":    "goodput_bps",
        "Retransmissions":       "retransmissions",
        "cwnd (K)":              "cwnd_k",
        "RTT (microsecond)":     "rtt_us",
        "RTT_var (microsecond)": "rttvar_us",
    }

    def to_csv_row(self, cols: Sequence[str]) -> Dict[str, Any]:
        raw = asdict(self)
        return {c: raw[self._MAP[c]] for c in cols if raw[self._MAP[c]] is not None}


# --------------------------------------------------------------------------- #
# Processor
# --------------------------------------------------------------------------- #

class IperfLogProcessor:
    """Parse one logfile and write/plot its CSV."""

    _OPT_ORDER: ClassVar[List[str]] = [
        "Retransmissions",
        "cwnd (K)",
        "RTT (microsecond)",
        "RTT_var (microsecond)",
    ]

    def __init__(self, path: Path, plot: bool = False) -> None:
        if not path.is_file():
            raise FileNotFoundError(path)
        self.path = path
        self.plot = plot
        self.logger = logging.getLogger(path.name)

    # public -----------------------------------------------------------------

    def run(self) -> Path:
        rows = self._parse()
        rows = self._trim_tail(rows)
        csv_p = self._write_csv(rows)
        if self.plot:
            self._plot(csv_p)
        return csv_p

    # internals --------------------------------------------------------------

    def _parse(self) -> List[IntervalRow]:
        rows: List[IntervalRow] = []
        with self.path.open(encoding="utf-8") as fh:
            for n, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    self.logger.debug("Line %d: bad JSON", n)
                    continue
                if rec.get("event") != "interval":
                    continue
                try:
                    st = rec["data"]["streams"][0]
                except (KeyError, IndexError):
                    self.logger.debug("Line %d: malformed interval", n)
                    continue

                rows.append(
                    IntervalRow(
                        start_time_sec=round(st.get("start", 0.0), 1),
                        end_time_sec=round(st.get("end",   0.0), 1),
                        goodput_bps=int(st.get("bits_per_second", 0)),
                        retransmissions=st.get("retransmits"),
                        cwnd_k=st.get("snd_cwnd", 0) /
                        1000 if "snd_cwnd" in st else None,
                        rtt_us=st.get("rtt"),
                        rttvar_us=st.get("rttvar"),
                    )
                )
        self.logger.info("Parsed %d rows", len(rows))
        return rows

    def _trim_tail(self, rows: List[IntervalRow]) -> List[IntervalRow]:
        if rows and rows[-1].end_time_sec - rows[-1].start_time_sec < 1.0:
            self.logger.info("Dropping sub‑second tail row")
            rows.pop()
        return rows

    def _write_csv(self, rows: List[IntervalRow]) -> Path:
        opt_present = {
            "Retransmissions" if any(
                r.retransmissions for r in rows) else None,
            "cwnd (K)" if any(r.cwnd_k for r in rows) else None,
            "RTT (microsecond)" if any(r.rtt_us for r in rows) else None,
            "RTT_var (microsecond)" if any(
                r.rttvar_us for r in rows) else None,
        } - {None}

        cols = [
            "start time (sec)",
            "end time   (sec)",
            "goodput (bits/sec)",
            *[f for f in self._OPT_ORDER if f in opt_present],
        ]

        csv_path = self.path.with_suffix(self.path.suffix + ".csv")
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow(r.to_csv_row(cols))
        self.logger.info("Wrote %s", csv_path)
        return csv_path

    def _plot(self, csv_path: Path) -> None:
        try:
            from iperf3_plotter import plot_csv
            self.logger.info("Plotting %s…", csv_path)
            plot_csv(str(csv_path))
        except ImportError:
            self.logger.error(
                "iperf3_plotter not found – ensure it is on PYTHONPATH")
        except Exception as e:  # noqa: BLE001
            self.logger.error("Plot failed: %s", e)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _config_log(v: int) -> None:
    lvl = logging.WARNING - min(v, 2) * 10  # 0→WARN 1→INFO 2→DEBUG
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(
        description="Convert iperf3 JSON‑L logs to CSV.")
    p.add_argument("log_files", nargs="+", help="iperf3 log files")
    p.add_argument("-v", action="count", default=0,
                   help="-v / -vv for INFO / DEBUG")
    p.add_argument("--plot", action="store_true",
                   help="plot CSVs after writing")
    args = p.parse_args(argv)

    _config_log(args.v)
    main_log = logging.getLogger("main")

    total_start = time.perf_counter()
    for f in args.log_files:
        start = time.perf_counter()
        try:
            IperfLogProcessor(Path(f), plot=args.plot).run()
            dur = time.perf_counter() - start
            main_log.info("Processed %s in %.3f s", f, dur)
        except Exception as err:  # noqa: BLE001
            main_log.error("%s: %s", f, err)
    total_dur = time.perf_counter() - total_start
    main_log.info("Finished %d file(s) in %.3f s",
                  len(args.log_files), total_dur)


if __name__ == "__main__":
    main(sys.argv[1:])
