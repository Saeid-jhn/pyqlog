"""qlog/sqlog parsers for picoquic and quiche.

Two on-disk formats are supported behind one interface:

* picoquic ``.qlog`` -- a single JSON document whose events are *positional
  arrays*: ``event[0]`` time (µs), ``event[1]`` category, ``event[2]`` type,
  ``event[3]`` data.
* quiche ``.sqlog`` -- JSON-SEQ: records separated by the ASCII record-separator
  char (``\\u001e``), each a JSON *object* keyed by ``name`` (e.g.
  ``transport:packet_sent``).

Time-unit normalization is the subtle part: picoquic timestamps are already in
microseconds, quiche timestamps are in milliseconds. Every parser emits ``time``
in **microseconds** so downstream code can divide by 1e6 uniformly. quiche RTT
metrics are likewise ms→µs scaled, and ``pacing_rate`` is bytes→bits (×8).
"""

from __future__ import annotations

import json
import logging
from enum import Enum
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# quiche record separator (JSON-SEQ / RFC 7464).
RECORD_SEPARATOR = "\x1e"

# quiche RTT metrics arrive in milliseconds and must be scaled to microseconds.
_RTT_KEYS = {"min_rtt", "smoothed_rtt", "latest_rtt", "rtt_variance"}


class QlogFormat(Enum):
    QLOG = ".qlog"
    SQLOG = ".sqlog"


# Tables produced by every qlog parser, in a stable order.
TABLE_NAMES: Tuple[str, ...] = (
    "packets", "metrics", "offsets", "datagram")


class BaseQlogParser:
    """Common logic for qlog parsers: direction mapping + DataFrame assembly."""

    def __init__(self, qlog_file: str):
        self.qlog_file = qlog_file

    # Subclasses implement the format-specific event walk.
    def extract(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        raise NotImplementedError

    @staticmethod
    def packet_direction(role: Optional[str]) -> Optional[str]:
        """Map a vantage-point role to the packet event we should count.

        Throughput is measured on the sending side, so a server log counts
        ``packet_sent`` and a client log counts ``packet_received``.
        """
        if role in ("server", "quiche-server qlog"):
            return "packet_sent"
        if role in ("client", "quiche-client qlog"):
            return "packet_received"
        logger.warning("Unrecognized qlog role: %s", role)
        return None

    @staticmethod
    def _build_dataframes(
        packets: List[dict], metrics: List[dict],
        offsets: List[dict], datagram: List[dict],
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df_packets = pd.DataFrame(packets)
        df_metrics = pd.DataFrame(metrics)
        df_offsets = pd.DataFrame(offsets)
        df_datagram = pd.DataFrame(datagram)

        # Flag retransmissions and accumulate on-wire bytes for the cumulative plot.
        if "packet_number" in df_packets.columns:
            df_packets["duplicate"] = df_packets.duplicated(
                subset=["packet_number"])
            df_packets["packet_size_cumsum"] = df_packets["packet_size"].cumsum()
        if "offset" in df_offsets.columns:
            df_offsets["duplicate"] = df_offsets.duplicated(subset=["offset"])

        return df_packets, df_metrics, df_offsets, df_datagram


class PicoquicParser(BaseQlogParser):
    """Parser for picoquic ``.qlog`` (single JSON document, array events)."""

    def extract(self):
        packets: List[dict] = []
        metrics: List[dict] = []
        offsets: List[dict] = []
        datagram: List[dict] = []

        with open(self.qlog_file, "r") as fh:
            doc = json.load(fh)
        trace = doc["traces"][0]
        role = trace["vantage_point"]["type"]
        direction = self.packet_direction(role)

        for event in trace["events"]:
            self._process_event(event, direction, packets,
                                 metrics, offsets, datagram)

        return self._build_dataframes(packets, metrics, offsets, datagram)

    @staticmethod
    def _process_event(event, direction, packets, metrics, offsets, datagram):
        time, category, etype, data = event[0], event[1], event[2], event[3]

        if category == "transport" and etype == direction:
            header = data["header"]
            packets.append({
                "time": time,
                "packet_number": header["packet_number"],
                "packet_size": header["packet_size"],
            })
            for frame in data["frames"]:
                if frame["frame_type"] == "stream":
                    offsets.append({
                        "time": time,
                        "offset": frame["offset"],
                        "length": frame["length"],
                    })
            datagram.append({
                "time": time,
                "length": header["packet_size"],
                "throughput": None,
            })

        elif category == "recovery" and etype == "metrics_updated":
            for key, value in data.items():
                metrics.append({"time": time, "key": key, "value": value})


class QuicheParser(BaseQlogParser):
    """Parser for quiche ``.sqlog`` (JSON-SEQ, object events; ms timestamps)."""

    def extract(self):
        packets: List[dict] = []
        metrics: List[dict] = []
        offsets: List[dict] = []
        datagram: List[dict] = []

        with open(self.qlog_file, "r") as fh:
            records = fh.read().split(RECORD_SEPARATOR)

        # The header record carries the role under 'title'.
        role = None
        for rec in records:
            rec = rec.strip()
            if rec:
                try:
                    role = json.loads(rec).get("title")
                except json.JSONDecodeError:
                    continue
                break
        direction = self.packet_direction(role)

        for rec in records:
            rec = rec.strip()
            if rec:
                self._process_record(
                    rec, direction, packets, metrics, offsets, datagram)

        return self._build_dataframes(packets, metrics, offsets, datagram)

    @staticmethod
    def _process_record(rec, direction, packets, metrics, offsets, datagram):
        try:
            ev = json.loads(rec)
        except json.JSONDecodeError:
            logger.warning("Skipping malformed sqlog record: %.80s...", rec)
            return

        time_us = ev.get("time", 0) * 1000  # ms -> µs

        if ev.get("name") == f"transport:{direction}":
            data = ev["data"]
            packets.append({
                "time": time_us,
                "packet_number": data["header"]["packet_number"],
                "packet_size": data["raw"]["length"],
            })
            for frame in data["frames"]:
                if frame["frame_type"] == "stream":
                    offsets.append({
                        "time": time_us,
                        "offset": frame["offset"],
                        "length": frame["length"],
                    })
            datagram.append({
                "time": time_us,
                "length": data["raw"]["length"],
                "throughput": None,
            })

        elif ev.get("name") == "recovery:metrics_updated":
            for key, raw in ev["data"].items():
                value = raw * 1000 if key in _RTT_KEYS else raw  # ms -> µs
                if key == "pacing_rate":
                    value *= 8  # bytes/s -> bits/s
                metrics.append({"time": time_us, "key": key, "value": value})


def get_parser(qlog_file: str) -> BaseQlogParser:
    """Return the right parser for ``qlog_file`` based on its extension."""
    if qlog_file.endswith(QlogFormat.QLOG.value):
        return PicoquicParser(qlog_file)
    if qlog_file.endswith(QlogFormat.SQLOG.value):
        return QuicheParser(qlog_file)
    raise ValueError(f"Unsupported qlog file format: {qlog_file}")
