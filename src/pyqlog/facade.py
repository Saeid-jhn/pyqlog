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
from typing import Dict, Iterable, List, Optional, Tuple

from .core import AnalysisResult
from .core.batch import run_many
from .registry import (
    get_analyzer_class, get_plotter_class, resolve_kind,
)

# A batch item for plot_many: a path plus the per-file kwargs plot() accepts.
PlotJob = Tuple[str, dict]

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


def _plot_job(job: "PlotJob") -> "Tuple[str, Optional[List[Path]], Optional[str]]":
    """Worker for :func:`plot_many`: plot one ``(path, kwargs)`` job.

    Module-level (so it pickles to a process pool) and never raises — one bad log
    must not kill the batch. Returns ``(path, image_paths, error)`` where exactly
    one of ``image_paths`` / ``error`` is set.
    """
    path, kwargs = job
    try:
        return (path, plot(path, **kwargs), None)
    except Exception as exc:  # keep one bad file from killing the batch
        return (path, None, str(exc))


def plot_many(
    jobs: "Iterable[PlotJob]",
    *,
    parallel: bool = True,
    workers: Optional[int] = None,
) -> "List[Tuple[str, Optional[List[Path]], Optional[str]]]":
    """Plot many artifacts concurrently, each with its own per-file options.

    Each job is a ``(path, kwargs)`` pair, where ``kwargs`` are the same keyword
    options :func:`plot` accepts (``out_dir``, ``formats``, ``save_csv``,
    ``kind``, and any analyzer/plotter opts such as pcap ``ports``). Because
    ``plot`` is single-file, this is the public entry point for callers that need
    to render a whole directory of logs in parallel (the ``pyqlog`` CLI plots
    multiple input files the same way).

    ``workers`` caps the process-pool size (default: one per CPU); ``parallel=
    False`` or a single job renders sequentially. Rendering is CPU-bound, so it
    scales well. Failures are captured per job rather than raised: returns a list
    of ``(path, image_paths, error)`` in input order, one per job.
    """
    return run_many(_plot_job, list(jobs), parallel=parallel, workers=workers)


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
