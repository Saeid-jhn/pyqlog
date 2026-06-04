# pyqlog

`pyqlog` parses and visualizes network-performance logs from three sources behind
one importable package and one CLI:

- **QUIC qlog** — picoquic `.qlog` and quiche `.sqlog`
- **iperf3** — JSON-L logs (`--json-stream`, optionally with `--get-server-output`)
- **pcap** — UDP/TCP/QUIC captures (via `tshark`)

It is designed to be driven both from the command line and as a library, so a
test-runner that produces logs can generate plots in one call — and re-plot later
from the saved data without rerunning the test.

## Requirements

- Python 3.9+
- numpy, pandas, matplotlib, seaborn (installed automatically)
- **pcap analysis only**: [TShark](https://www.wireshark.org/) on your `PATH`
  (`sudo apt install tshark` / `brew install wireshark`)

## Install

```bash
pip install -e .          # runtime
pip install -e ".[test]"  # + pytest for the test suite
```

This installs a single `pyqlog` console command.

## Library usage

```python
import pyqlog

# Analyze + plot in one call; log type is auto-detected from the extension.
# Intermediate CSVs are written next to the plots by default.
pyqlog.plot("run1.sqlog", out_dir="results", formats=["png", "pdf"])

# Just the data (DataFrames in memory, no disk writes):
result = pyqlog.analyze("run1.sqlog")
result.tables["data_rate"]        # pandas DataFrame
result.to_csv("results/run1")     # persist for later

# Re-plot later from the saved CSVs — no original log, no re-analysis:
pyqlog.replot("results/run1.sqlog", formats=["svg"])
```

For full control, use the explicit classes:

```python
from pyqlog.qlog import QlogAnalyzer, QlogPlotter
from pyqlog.iperf import IperfAnalyzer, IperfPlotter
from pyqlog.pcap import PcapAnalyzer, PcapPlotter

result = QlogAnalyzer("run1.sqlog", interval="500ms").analyze()
QlogPlotter(result).render(out_dir="results", formats=["png"])
```

Every analyzer returns an `AnalysisResult` (named DataFrames + metadata) that any
matching plotter consumes, and that round-trips through `to_csv`/`from_csv`.

## CLI usage

```bash
# QUIC qlog/sqlog
pyqlog qlog file.qlog file2.sqlog --interval 1000ms --formats png svg

# iperf3
pyqlog iperf log.json --title -m rtt cwnd --formats png

# pcap (needs tshark)
pyqlog pcap capture.pcap --tcp --port 443 5201 --total --sequence --tcp-error

# auto-detect log type
pyqlog plot some.sqlog another.json --out-dir results

# re-plot from previously saved CSVs (no original log needed)
pyqlog replot results/run1.sqlog --formats pdf
```

Use `-v`/`-vv` for INFO/DEBUG logging. Multiple input files are processed in
parallel. `--no-csv` skips writing the intermediate CSVs.

## How it works

| Stage | What happens |
|-------|--------------|
| **Analyze** | A per-format parser reads the log into an `AnalysisResult` — named DataFrames plus run metadata. QUIC timestamps are normalized to microseconds (quiche ms→µs, RTT ms→µs, pacing bytes→bits); throughput is on-wire datagram bytes while goodput counts only non-duplicate stream offsets. |
| **Persist** | `AnalysisResult.to_csv(prefix)` writes one `{prefix}.{table}.csv` per table plus `{prefix}.meta.json`. |
| **Plot / Replot** | A plotter renders the result to PNG/SVG/PDF. `replot` reloads the CSVs via `from_csv`, so figures can be re-rendered without the original log or re-analysis. |

## Tests

```bash
pytest
```

The suite uses tiny synthetic fixtures (no external capture files); the pcap test
is skipped automatically when `tshark` is not installed.
