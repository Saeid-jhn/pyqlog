"""iperf3 JSON-L log analyzer.

Handles plain sender logs and client logs captured with
``iperf3 --get-server-output`` (a single file holding both sender- and
receiver-side intervals). Sender, receiver and embedded server intervals are
merged per ``(start, end)`` window into one tidy ``intervals`` table; per-test
metadata is collected separately into ``AnalysisResult.metadata``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from ..core import AnalysisResult

logger = logging.getLogger(__name__)

# Default display timezone for the human-readable timestamp. iperf reports GMT;
# the original tool hard-coded CEST (+2h). Kept as the default, now overridable.
DEFAULT_TZ = timezone(timedelta(hours=2))
DEFAULT_TZ_LABEL = "CEST"

INTERVAL_COLS: List[str] = [
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

META_COLS: List[str] = [
    "local_host",
    "remote_host",
    "remote_port",
    "version",
    "timestamp",
    "tcp_mss_default",
    "sndbuf_actual",
    "rcvbuf_actual",
    "test_start.protocol",
    "test_start.bytes",
    "test_start.reverse",
]


@dataclass(slots=True)
class IntervalRow:
    start_time_sec: float
    end_time_sec: float
    rcv_goodput_bps: Optional[int] = None
    snd_throughput_bps: Optional[int] = None
    retransmits: Optional[int] = None
    snd_cwnd_k: Optional[float] = None
    snd_wnd_k: Optional[float] = None
    rtt_us: Optional[int] = None
    rtt_var_us: Optional[int] = None

    _MAP: ClassVar[Dict[str, str]] = {
        "start_time (s)": "start_time_sec",
        "end_time (s)": "end_time_sec",
        "rcv_goodput (bps)": "rcv_goodput_bps",
        "snd_throughput (bps)": "snd_throughput_bps",
        "retransmits": "retransmits",
        "snd_cwnd (K)": "snd_cwnd_k",
        "snd_wnd (K)": "snd_wnd_k",
        "rtt (us)": "rtt_us",
        "rtt_var (us)": "rtt_var_us",
    }

    def as_record(self) -> Dict[str, Any]:
        raw = asdict(self)
        return {col: raw[attr] for col, attr in self._MAP.items()}


class IperfAnalyzer:
    """Parse one iperf3 JSON-L file into an :class:`AnalysisResult`."""

    def __init__(self, path: str, *, tz: timezone = DEFAULT_TZ,
                 tz_label: str = DEFAULT_TZ_LABEL):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Log file not found: {self.path}")
        self.tz = tz
        self.tz_label = tz_label
        self.metadata: Dict[str, Any] = {}

    def analyze(self) -> AnalysisResult:
        rows = self._parse()
        rows = self._drop_subsecond_tail(rows)

        df = pd.DataFrame([r.as_record() for r in rows], columns=INTERVAL_COLS)
        return AnalysisResult(
            kind="iperf",
            tables={"intervals": df},
            metadata=self.metadata,
            source=str(self.path),
        )

    # -- parsing ---------------------------------------------------------- #

    def _parse(self) -> List[IntervalRow]:
        rows_by_interval: Dict[Tuple[float, float], IntervalRow] = {}

        with self.path.open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    logger.debug("line %d: invalid JSON", lineno)
                    continue

                if rec.get("event") == "start" and not self.metadata:
                    self.metadata = self._extract_metadata(rec["data"])
                    continue

                if rec.get("event") == "interval":
                    blocks = [rec["data"]]
                elif rec.get("event") == "server_output_json":
                    blocks = rec["data"].get("intervals", [])
                else:
                    continue

                for blk in blocks:
                    self._merge_block(blk, rows_by_interval)

        logger.info("parsed %d interval rows", len(rows_by_interval))
        return sorted(rows_by_interval.values(), key=lambda r: r.start_time_sec)

    def _extract_metadata(self, d: Dict[str, Any]) -> Dict[str, Any]:
        conn = d.get("connected", [{}])[0]
        t = datetime.strptime(
            d["timestamp"]["time"], "%a, %d %b %Y %H:%M:%S GMT"
        ).replace(tzinfo=timezone.utc).astimezone(self.tz)
        ts = t.strftime(f"%Y-%m-%d %H:%M:%S {self.tz_label}")
        return {
            "local_host": conn.get("local_host"),
            "remote_host": conn.get("remote_host"),
            "remote_port": conn.get("remote_port"),
            "version": d.get("version"),
            "timestamp": ts,
            "tcp_mss_default": d.get("tcp_mss_default"),
            "sndbuf_actual": d.get("sndbuf_actual"),
            "rcvbuf_actual": d.get("rcvbuf_actual"),
            "test_start.protocol": d["test_start"].get("protocol"),
            "test_start.bytes": d["test_start"].get("bytes"),
            "test_start.reverse": d["test_start"].get("reverse"),
        }

    @staticmethod
    def _merge_block(blk: Dict[str, Any],
                     rows_by_interval: Dict[Tuple[float, float], IntervalRow]) -> None:
        streams = blk.get("streams", [])
        if not streams:
            return
        st = streams[0]

        beg = round(st.get("start", 0.0), 3)
        end = round(st.get("end", 0.0), 3)
        row = rows_by_interval.setdefault((beg, end), IntervalRow(beg, end))

        bps = int(st.get("bits_per_second", 0))
        if st.get("sender", True):
            row.snd_throughput_bps = bps
            row.retransmits = st.get("retransmits") or row.retransmits
            if "snd_cwnd" in st:
                row.snd_cwnd_k = st["snd_cwnd"] / 1000
            if "snd_wnd" in st:
                row.snd_wnd_k = st["snd_wnd"] / 1000
            row.rtt_us = st.get("rtt") or row.rtt_us
            row.rtt_var_us = st.get("rttvar") or row.rtt_var_us
        else:
            row.rcv_goodput_bps = bps

    def _drop_subsecond_tail(self, rows: List[IntervalRow]) -> List[IntervalRow]:
        if rows and (rows[-1].end_time_sec - rows[-1].start_time_sec < 1.0):
            logger.info("dropping final sub-second row %s-%s",
                        rows[-1].start_time_sec, rows[-1].end_time_sec)
            rows.pop()
        return rows
