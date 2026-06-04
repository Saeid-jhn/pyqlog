"""The :class:`AnalysisResult` data contract shared by all log types.

An ``AnalysisResult`` is the single object that flows from an analyzer to a
plotter. It holds one or more named DataFrames plus scalar run metadata, and it
knows how to persist itself to CSV and reload from CSV. This is what makes the
"analyze once, re-plot later without rerunning the test" workflow possible: the
saved CSVs are a complete, replottable snapshot.
"""

from __future__ import annotations

import glob
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_META_SUFFIX = ".meta.json"


@dataclass
class AnalysisResult:
    """Container for the output of an analyzer.

    Attributes:
        kind:      Log family, one of ``"qlog"`` / ``"iperf"`` / ``"pcap"``.
        tables:    Named DataFrames (e.g. ``{"metrics": df, "data_rate": df}``).
        metadata:  Scalar run info (iperf header fields, qlog role, ...).
        source:    Original log path, when known.
    """

    kind: str
    tables: Dict[str, pd.DataFrame] = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)
    source: Optional[str] = None

    # -- convenience ------------------------------------------------------ #

    def table(self, name: str) -> pd.DataFrame:
        """Return a table, or an empty DataFrame if it is missing."""
        return self.tables.get(name, pd.DataFrame())

    def is_empty(self) -> bool:
        """True when every table is empty (nothing worth plotting)."""
        return all(df.empty for df in self.tables.values())

    # -- persistence ------------------------------------------------------ #

    def to_csv(self, prefix: str) -> List[Path]:
        """Write each table to ``{prefix}.{name}.csv`` and metadata to JSON.

        ``prefix`` may include a directory; parent directories are created.
        Returns the paths written.
        """
        prefix = str(prefix)
        parent = os.path.dirname(prefix)
        if parent:
            os.makedirs(parent, exist_ok=True)

        written: List[Path] = []
        for name, df in self.tables.items():
            out = Path(f"{prefix}.{name}.csv")
            df.to_csv(out, index=False)
            written.append(out)

        meta_payload = {"kind": self.kind, "source": self.source,
                        "metadata": self.metadata}
        meta_path = Path(f"{prefix}{_META_SUFFIX}")
        meta_path.write_text(json.dumps(meta_payload, indent=2, default=str))
        written.append(meta_path)

        logger.info("Wrote %d artifact(s) for prefix %s", len(written), prefix)
        return written

    @classmethod
    def from_csv(cls, prefix: str, kind: Optional[str] = None) -> "AnalysisResult":
        """Reconstruct a result from CSVs previously written by :meth:`to_csv`.

        Globs ``{prefix}.*.csv`` to recover every table; ``{prefix}.meta.json``
        supplies ``kind``/``metadata``/``source`` when present. ``kind`` can be
        passed explicitly to override or supply it when no metadata file exists.
        """
        prefix = str(prefix)
        tables: Dict[str, pd.DataFrame] = {}
        for path in sorted(glob.glob(f"{prefix}.*.csv")):
            # name is the segment between the prefix and the .csv extension
            name = Path(path).name[len(Path(prefix).name) + 1: -len(".csv")]
            tables[name] = pd.read_csv(path)

        metadata: Dict = {}
        source: Optional[str] = None
        meta_path = Path(f"{prefix}{_META_SUFFIX}")
        if meta_path.exists():
            payload = json.loads(meta_path.read_text())
            kind = kind or payload.get("kind")
            metadata = payload.get("metadata", {})
            source = payload.get("source")

        if kind is None:
            raise ValueError(
                f"Cannot determine 'kind' for prefix {prefix!r}; pass kind= "
                "or ensure a .meta.json exists."
            )
        if not tables:
            raise FileNotFoundError(f"No CSV tables found for prefix {prefix!r}")

        return cls(kind=kind, tables=tables, metadata=metadata, source=source)
