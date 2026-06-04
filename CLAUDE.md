# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`pyqlog` is a single importable package (`src/pyqlog/`) that parses and visualizes
three kinds of network-performance logs behind one API and one CLI:

- **qlog** — QUIC qlog parser/plotter (picoquic `.qlog`, quiche `.sqlog`)
- **iperf** — iperf3 JSON-L logs
- **pcap** — UDP/TCP/QUIC captures (wraps `tshark`)

The three were previously independent top-level scripts under `src/{qlog,pcap,iperflog}/`;
they are now subpackages of `pyqlog` sharing a common core.

## Commands

```bash
pip install -e .            # runtime install (provides the `pyqlog` console command)
pip install -e ".[test]"    # + pytest

pytest                      # run the test suite (synthetic fixtures; pcap test skipped w/o tshark)

# CLI (all subcommands accept multiple files, processed in parallel)
pyqlog qlog  file.qlog file2.sqlog --interval 1000ms --formats png svg [--no-csv]
pyqlog iperf log.json --title -m rtt cwnd --formats png
pyqlog pcap  capture.pcap --tcp --port 443 --total --sequence --tcp-error   # needs tshark
pyqlog plot  any.sqlog any.json --out-dir results      # auto-detect by extension
pyqlog replot results/run.sqlog --formats pdf          # re-render from saved CSVs only
```

Local dev note: the working environment is PEP 668 "externally managed". Use a venv
created with `python3 -m venv --system-site-packages .venv` (reuses system
pandas/matplotlib), then `.venv/bin/pip install -e ".[test]"`. There is no linter config.

## Architecture

### One data contract: `AnalysisResult` (`core/result.py`)
Every analyzer returns an `AnalysisResult(kind, tables, metadata, source)` where `tables`
is a dict of named DataFrames. Plotters consume it directly (in memory). `to_csv(prefix)`
writes `{prefix}.{table}.csv` per table plus `{prefix}.meta.json`; `from_csv(prefix)` globs
those back. This is what enables **replot without rerunning**: `facade.replot` reloads the
CSVs and renders, needing neither the original log nor re-analysis.

### Layered API
- `facade.py` — `analyze()`, `plot()`, `replot()`. `plot` = analyze → (optional `to_csv`)
  → render, all in memory. Flat keyword options are routed to the right analyzer/plotter
  by `kind` via the `_ANALYZER_OPTS`/`_PLOTTER_OPTS` allow-lists.
- `registry.py` — extension→kind detection (`.qlog`/`.sqlog`→qlog, `.json`/`.jsonl`→iperf,
  `.pcap`/`.pcapng`→pcap) and kind→(Analyzer, Plotter) lookup. `kind=` overrides detection.
- `cli.py` — thin argparse wrappers over the facade; batch parallelism via `core/batch.run_many`.

When adding a new log type: add a subpackage with an `Analyzer.analyze()->AnalysisResult`
and a `Plotter(result).render(...)`, then register both in `registry.py`.

### Shared core (`core/`)
- `units.auto_scale` — the single unit-scaling helper (previously duplicated in all 3 plotters).
- `plotting.apply_style` / `plotting.save_figure` — shared theme + per-format saving (PNG
  defaults to `PNG_DPI = 300`).
- `batch.run_many` — multiprocessing fan-out (sequential for a single file / when `parallel=False`).

### qlog specifics (`qlog/parsers.py`, `qlog/analyzer.py`, `qlog/plotter.py`)
- Two formats, one `BaseQlogParser` interface: `PicoquicParser` (`.qlog`, single JSON doc,
  events are positional **arrays** `event[0..3]`) and `QuicheParser` (`.sqlog`, JSON-SEQ split
  on the ASCII record separator `\x1e`, events are **objects** keyed by `name`).
- **Time-unit normalization (easy to get wrong):** picoquic timestamps are already µs; quiche
  `time` is ms (×1000→µs), quiche RTT keys (`min_rtt`/`smoothed_rtt`/`latest_rtt`/`rtt_variance`)
  ×1000, and `pacing_rate` ×8 (bytes→bits). Downstream divides `time` by 1e6 for seconds.
- **Direction by vantage point:** server logs count `packet_sent`, client logs count
  `packet_received` (`BaseQlogParser.packet_direction`).
- **Throughput vs goodput** (`compute_data_rate`): throughput sums datagram (on-wire) bytes;
  goodput sums only non-duplicate stream-frame offset bytes; both binned with `np.histogram`
  over the `--interval` window. Tables: `packets`, `metrics`, `offsets`, `datagram`, `data_rate`.

### iperf specifics (`iperf/analyzer.py`)
Merges sender / receiver / embedded `server_output_json` intervals keyed by `(start,end)`;
drops the final sub-second interval; emits the tidy `intervals` table with `INTERVAL_COLS`
order and run header in `metadata`. The GMT→display-timezone conversion defaults to CEST
(+2h) but is configurable via `IperfAnalyzer(tz=, tz_label=)`.

### pcap specifics (`pcap/analyzer.py`)
Runs `tshark` once (`pcap_to_df`) extracting `FIELDS`, coalesces tcp/udp into `SrcPort`/`DstPort`,
maps proto, derives per-port + total throughput and TCP error counts. Port *filtering* is
deferred to `PcapPlotter` so one analysis can be re-plotted for different port subsets. Tables:
`throughput`, `sequence`. `tshark_available()` gates the pcap path.
