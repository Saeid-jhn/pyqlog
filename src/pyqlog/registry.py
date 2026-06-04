"""Map log files to the right analyzer/plotter by ``kind`` or file extension."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Type

from .core import AnalysisResult
from .qlog import QlogAnalyzer, QlogPlotter
from .iperf import IperfAnalyzer, IperfPlotter
from .pcap import PcapAnalyzer, PcapPlotter

# Extension -> kind. ``.qlog``/``.sqlog`` both map to the qlog family; the
# qlog parser picks picoquic vs quiche from the extension internally.
_EXT_TO_KIND: Dict[str, str] = {
    ".qlog": "qlog",
    ".sqlog": "qlog",
    ".json": "iperf",
    ".jsonl": "iperf",
    ".pcap": "pcap",
    ".pcapng": "pcap",
}

# kind -> (Analyzer, Plotter)
_KIND_TO_CLASSES: Dict[str, Tuple[Type, Type]] = {
    "qlog": (QlogAnalyzer, QlogPlotter),
    "iperf": (IperfAnalyzer, IperfPlotter),
    "pcap": (PcapAnalyzer, PcapPlotter),
}


def detect_kind(path: str) -> str:
    """Infer the log ``kind`` from a file's extension."""
    ext = Path(path).suffix.lower()
    kind = _EXT_TO_KIND.get(ext)
    if kind is None:
        raise ValueError(
            f"Cannot detect log kind from extension {ext!r} (path: {path}). "
            f"Pass kind= explicitly. Known: {sorted(set(_EXT_TO_KIND.values()))}."
        )
    return kind


def get_analyzer_class(kind: str) -> Type:
    return _KIND_TO_CLASSES[kind][0]


def get_plotter_class(kind: str) -> Type:
    return _KIND_TO_CLASSES[kind][1]


def resolve_kind(path: str, kind: Optional[str]) -> str:
    """Return ``kind`` if given, else detect it from ``path``."""
    if kind is not None:
        if kind not in _KIND_TO_CLASSES:
            raise ValueError(
                f"Unknown kind {kind!r}; known: {sorted(_KIND_TO_CLASSES)}.")
        return kind
    return detect_kind(path)
