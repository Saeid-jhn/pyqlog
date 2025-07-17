#!/usr/bin/env python3
from __future__ import annotations

"""
iperf_log_processor.py – convert iperf3 JSON-line logs to CSV.

Usage:
    python iperf_log_processor.py [-v|-vv] <log1> [log2 ...]
"""

import argparse
import csv
import json
import logging
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

    _FIELD_MAP: ClassVar[Dict[str, str]] = {
        "start time (sec)":     "start_time_sec",
        "end time (sec)":       "end_time_sec",
        "goodput (bits/sec)":   "goodput_bps",
        "Retransmissions":      "retransmissions",
        "cwnd (K)":             "cwnd_k",
        "RTT (microsecond)":    "rtt_us",
        "RTT_var (microsecond)": "rttvar_us",
    }

    def to_csv_row(self, columns: Sequence[str]) -> Dict[str, Any]:
        """Return a dict restricted to *columns* order."""
        raw = asdict(self)
        return {col: raw[self._FIELD_MAP[col]] for col in columns if raw[self._FIELD_MAP[col]] is not None}


# --------------------------------------------------------------------------- #
# Processor
# --------------------------------------------------------------------------- #

class IperfLogProcessor:
    """Parse one iperf3 logfile and write its CSV counterpart."""

    _OPTIONAL_ORDER: ClassVar[List[str]] = [
        "Retransmissions",
        "cwnd (K)",
        "RTT (microsecond)",
        "RTT_var (microsecond)",
    ]

    def __init__(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(path)
        self.path = path
        self.logger = logging.getLogger(path.name)

    # public -----------------------------------------------------------------

    def run(self) -> Path:
        rows = self._parse()
        rows = self._trim_tail(rows)
        csv_path = self._write_csv(rows)
        self.logger.info("Saved %s", csv_path)
        return csv_path

    # internals ---------------------------------------------------------------

    def _parse(self) -> List[IntervalRow]:
        rows: List[IntervalRow] = []
        with self.path.open(encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    self.logger.debug("Line %d: invalid JSON", line_no)
                    continue
                if rec.get("event") != "interval":
                    continue

                try:
                    st = rec["data"]["streams"][0]
                except (KeyError, IndexError):
                    self.logger.debug(
                        "Line %d: malformed interval record", line_no)
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
            self.logger.info("Dropping sub-second tail row")
            rows.pop()
        return rows

    def _write_csv(self, rows: List[IntervalRow]) -> Path:
        optional_present = {
            "Retransmissions" if any(
                r.retransmissions is not None for r in rows) else None,
            "cwnd (K)" if any(r.cwnd_k is not None for r in rows) else None,
            "RTT (microsecond)" if any(
                r.rtt_us is not None for r in rows) else None,
            "RTT_var (microsecond)"if any(
                r.rttvar_us is not None for r in rows) else None,
        } - {None}

        columns = [
            "start time (sec)",
            "end time (sec)",
            "goodput (bits/sec)",
            *[f for f in self._OPTIONAL_ORDER if f in optional_present],
        ]

        csv_path = self.path.with_suffix(self.path.suffix + ".csv")
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns)
            writer.writeheader()
            for r in rows:
                writer.writerow(r.to_csv_row(columns))

        return csv_path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _configure_logging(verbosity: int) -> None:
    level = logging.WARNING - min(verbosity, 2) * 10  # 0→WARN, 1→INFO, 2→DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(
        description="Convert iperf3 JSON-line logs to CSV.")
    p.add_argument("log_files", nargs="+", metavar="LOG",
                   help="iperf3 log files")
    p.add_argument("-v", action="count", default=0,
                   help="-v / -vv for INFO / DEBUG")
    args = p.parse_args(argv)

    _configure_logging(args.v)

    for fname in args.log_files:
        try:
            IperfLogProcessor(Path(fname)).run()
        except Exception as err:  # noqa: BLE001
            logging.error("%s: %s", fname, err)


if __name__ == "__main__":
    main()
