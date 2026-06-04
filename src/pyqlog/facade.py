"""One-call API: ``analyze``, ``plot`` and ``replot`` with log-type auto-detect.

This is the entry point an external test-runner should use::

    import pyqlog
    pyqlog.plot("run1.sqlog", out_dir="results")          # analyze + plot
    pyqlog.replot("results/run1.sqlog", formats=["pdf"])  # re-plot from saved CSVs

``plot`` keeps everything in memory (no forced disk round-trip) and, by default,
also writes the intermediate CSVs so the run can be re-plotted later with
``replot`` without touching the original log.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .core import AnalysisResult
from .registry import (
    get_analyzer_class, get_plotter_class, resolve_kind,
)

# Which constructor kwargs each kind's analyzer / plotter accepts. Used to route
# the flat facade options to the right class without leaking irrelevant ones.
_ANALYZER_OPTS: Dict[str, set] = {
    "qlog": {"interval"},
    "iperf": {"tz", "tz_label"},
    "pcap": {"interval", "stream_index", "tcp_only", "quic_only"},
}
_PLOTTER_OPTS: Dict[str, set] = {
    "qlog": set(),
    "iperf": {"title", "metrics"},
    "pcap": {"ports", "legends", "total", "sequence", "tcp_error"},
}


def _filter(opts: dict, allowed: set) -> dict:
    return {k: v for k, v in opts.items() if k in allowed and v is not None}


def analyze(path: str, *, kind: Optional[str] = None, **opts) -> AnalysisResult:
    """Parse ``path`` into an :class:`AnalysisResult` (no plotting, no disk writes)."""
    kind = resolve_kind(path, kind)
    analyzer_cls = get_analyzer_class(kind)
    return analyzer_cls(path, **_filter(opts, _ANALYZER_OPTS[kind])).analyze()


def _default_prefix(source: Optional[str], out_dir: Optional[str]) -> str:
    base = os.path.basename(source) if source else "output"
    return os.path.join(out_dir, base) if out_dir else base


def plot(
    path: str,
    *,
    out_dir: Optional[str] = None,
    formats: Iterable[str] = ("png",),
    kind: Optional[str] = None,
    save_csv: bool = True,
    csv_prefix: Optional[str] = None,
    **opts,
) -> List[Path]:
    """Analyze ``path`` and render plots, optionally persisting CSVs for replot.

    Extra keyword options are routed to the relevant analyzer/plotter by ``kind``
    (e.g. ``interval=`` for qlog/pcap, ``ports=``/``total=`` for pcap,
    ``title=``/``metrics=`` for iperf). Returns the image paths written.
    """
    kind = resolve_kind(path, kind)
    result = analyze(path, kind=kind, **opts)

    prefix = csv_prefix or _default_prefix(result.source, out_dir)
    if save_csv:
        result.to_csv(prefix)

    return _render(result, kind, out_dir=out_dir, formats=formats,
                   prefix=prefix, opts=opts)


def replot(
    csv_prefix: str,
    *,
    kind: Optional[str] = None,
    out_dir: Optional[str] = None,
    formats: Iterable[str] = ("png",),
    **opts,
) -> List[Path]:
    """Re-render plots from CSVs previously written by :meth:`AnalysisResult.to_csv`.

    No original log is needed. ``out_dir`` defaults to the CSVs' own directory.
    """
    result = AnalysisResult.from_csv(csv_prefix, kind=kind)
    if out_dir is None:
        out_dir = os.path.dirname(csv_prefix) or None
    prefix = csv_prefix
    return _render(result, result.kind, out_dir=out_dir, formats=formats,
                   prefix=os.path.basename(prefix) if out_dir else prefix,
                   opts=opts)


def _render(result: AnalysisResult, kind: str, *, out_dir, formats, prefix,
            opts: dict) -> List[Path]:
    plotter_cls = get_plotter_class(kind)
    plotter = plotter_cls(result, **_filter(opts, _PLOTTER_OPTS[kind]))
    return plotter.render(out_dir=out_dir, formats=formats,
                          prefix=os.path.basename(prefix))
