"""Unified ``pyqlog`` command-line interface.

Subcommands:
  qlog    parse + plot QUIC qlog/sqlog files
  iperf   parse + plot iperf3 JSON-L logs
  pcap    parse + plot pcap captures (needs tshark)
  plot    auto-detect log type and plot
  replot  re-render plots from previously saved CSVs (no original log needed)
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import List, Optional, Sequence

from .core.batch import run_many
from . import facade


def _configure_logging(verbosity: int) -> None:
    level = logging.WARNING - min(verbosity, 2) * 10
    logging.basicConfig(level=level,
                        format="%(asctime)s | %(levelname)-8s | %(message)s",
                        datefmt="%H:%M:%S")


# -- batch workers (module-level so multiprocessing can pickle them) -------- #

def _plot_worker(path: str, kwargs: dict) -> List:
    try:
        return facade.plot(path, **kwargs)
    except Exception:  # keep one bad file from killing the batch
        logging.exception("Failed to process %s", path)
        return []


def _replot_worker(prefix: str, kwargs: dict) -> List:
    try:
        return facade.replot(prefix, **kwargs)
    except Exception:
        logging.exception("Failed to replot %s", prefix)
        return []


def _run_plot(files: Sequence[str], kwargs: dict, parallel: bool = True) -> None:
    run_many(_plot_worker, list(files), extra_args=(kwargs,), parallel=parallel)


# -- argument parser --------------------------------------------------------- #

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pyqlog",
                                description="Parse & visualize QUIC qlog, pcap and iperf3 logs.")
    p.add_argument("-v", "--verbose", action="count", default=0,
                   help="-v for INFO, -vv for DEBUG")
    sub = p.add_subparsers(dest="command", required=True)

    common_fmt = dict(nargs="+", metavar="FMT",
                      choices=["png", "svg", "pdf"], default=["png"])

    # qlog
    q = sub.add_parser("qlog", help="parse + plot QUIC qlog/sqlog files")
    q.add_argument("files", nargs="+")
    q.add_argument("--interval", default="1000ms",
                   help="binning interval (default: 1000ms)")
    q.add_argument("--formats", **common_fmt)
    q.add_argument("--no-csv", action="store_true", help="don't write CSVs")
    q.add_argument("--out-dir", default=None)

    # iperf
    i = sub.add_parser("iperf", help="parse + plot iperf3 JSON-L logs")
    i.add_argument("files", nargs="+")
    i.add_argument("--title", action="store_true", help="show metadata title")
    i.add_argument("-m", "--metrics", nargs="+", metavar="METRIC",
                   help="only plot these metrics")
    i.add_argument("--formats", **common_fmt)
    i.add_argument("--no-csv", action="store_true")
    i.add_argument("--out-dir", default=None)

    # pcap
    c = sub.add_parser("pcap", help="parse + plot pcap captures (needs tshark)")
    c.add_argument("files", nargs="+")
    c.add_argument("--interval", type=float, default=1.0)
    c.add_argument("--stream-index", type=int)
    c.add_argument("--tcp", action="store_true")
    c.add_argument("--quic", action="store_true")
    c.add_argument("--port", type=int, nargs="+")
    c.add_argument("--port-legend", type=str, nargs="+")
    c.add_argument("--total", action="store_true")
    c.add_argument("--sequence", action="store_true")
    c.add_argument("--tcp-error", action="store_true")
    c.add_argument("--formats", **common_fmt)
    c.add_argument("--no-csv", action="store_true")
    c.add_argument("--out-dir", default=None)

    # plot (auto-detect)
    pl = sub.add_parser("plot", help="auto-detect log type and plot")
    pl.add_argument("files", nargs="+")
    pl.add_argument("--kind", choices=["qlog", "iperf", "pcap"])
    pl.add_argument("--formats", **common_fmt)
    pl.add_argument("--no-csv", action="store_true")
    pl.add_argument("--out-dir", default=None)

    # replot
    rp = sub.add_parser("replot", help="re-plot from saved CSVs")
    rp.add_argument("prefixes", nargs="+", help="CSV prefixes (no extension)")
    rp.add_argument("--kind", choices=["qlog", "iperf", "pcap"])
    rp.add_argument("--formats", **common_fmt)
    rp.add_argument("--out-dir", default=None)

    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    _configure_logging(args.verbose)

    if args.command == "qlog":
        _run_plot(args.files, dict(kind="qlog", interval=args.interval,
                                   formats=args.formats, save_csv=not args.no_csv,
                                   out_dir=args.out_dir))

    elif args.command == "iperf":
        _run_plot(args.files, dict(kind="iperf", title=args.title,
                                   metrics=args.metrics, formats=args.formats,
                                   save_csv=not args.no_csv, out_dir=args.out_dir))

    elif args.command == "pcap":
        legends = None
        if args.port:
            legends = (dict(zip(args.port, args.port_legend))
                       if args.port_legend else {p: f"Port {p}" for p in args.port})
        _run_plot(args.files, dict(kind="pcap", interval=args.interval,
                                   stream_index=args.stream_index,
                                   tcp_only=args.tcp, quic_only=args.quic,
                                   ports=args.port, legends=legends,
                                   total=args.total, sequence=args.sequence,
                                   tcp_error=args.tcp_error, formats=args.formats,
                                   save_csv=not args.no_csv, out_dir=args.out_dir))

    elif args.command == "plot":
        _run_plot(args.files, dict(kind=args.kind, formats=args.formats,
                                   save_csv=not args.no_csv, out_dir=args.out_dir))

    elif args.command == "replot":
        run_many(_replot_worker, list(args.prefixes),
                 extra_args=(dict(kind=args.kind, formats=args.formats,
                                  out_dir=args.out_dir),))


if __name__ == "__main__":
    main(sys.argv[1:])
