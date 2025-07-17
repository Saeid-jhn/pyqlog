# pcap_analyzer.py

Comprehensive PCAP analzyer for UDP, TCP, and QUIC traffic analysis.  
Generates throughput plots, optional TCP‐sequence scatter plots, TCP error bar charts, and flexible exports (PNG, CSV, SVG, PDF).

---

## Table of Contents

1. [Features](#features)  
2. [Requirements](#requirements)  
3. [Installation](#installation)  
4. [Usage](#usage)  
5. [CLI Options](#cli-options)  
6. [Examples](#examples)  
7. [Output Files](#output-files)  
8. [Customization](#customization)   

---

## Features

- **Throughput per port** and **total throughput** (in Mbps) over configurable time intervals.
- **TCP sequence number** scatter plot  
- **TCP error events** (retransmissions, out‑of‑order, fast retransmissions) bar chart  
- Heuristic QUIC detection on common QUIC ports  
- Multi‑format export: PNG, CSV, SVG, PDF  
- Parallel processing of multiple PCAPs  

---

## Requirements

- Python 3.8+  
- [TShark](https://www.wireshark.org/docs/man-pages/tshark.html) (Wireshark CLI) in your `PATH`  
  - **Debian/Ubuntu**  
    ```bash
    sudo apt update
    sudo apt install tshark
    ```
  - **macOS** (with Homebrew)  
    ```bash
    brew install wireshark
    ```
  - **Windows**  
    1. Download the Wireshark installer from https://www.wireshark.org/  
    2. During installation, ensure “TShark” (the CLI tools) is selected.  
    3. Add the install directory (e.g. `C:\Program Files\Wireshark`) to your `PATH`.  
- Python packages:
    ```bash 
    pip install pandas seaborn matplotlib 
    ```

---

## Installation

1. Clone or download this repository  
2. Ensure `tshark` is installed and accessible:  
    ```bash
    which tshark       # Linux/macOS
    where tshark       # Windows PowerShell
    tshark -v
    ```
## Usage

```bash
./pcap_analyzer.py [OPTIONS] <pcap_file> [<pcap_file>...]
```

---

## CLI Options

| Option                      | Description                                                                 |
|-----------------------------|-----------------------------------------------------------------------------|
| `<pcap_files>`              | One or more input PCAP files                                                  |
| `--interval FLOAT`          | Bin width in seconds for throughput aggregation (default: 1.0)                |
| `--stream-index INT`        | Only process a specific TCP stream index                                     |
| `--tcp`                     | Analyze only TCP traffic                                                      |
| `--quic`                    | Analyze only QUIC traffic           |
| `--port INT [INT ...]`      | List of source ports to plot separately                                       |
| `--port-legend TEXT [TEXT …]` | Custom legend labels corresponding to `--port` entries                      |
| `--total`                   | Include total throughput line                                                 |
| `--sequence`                | Generate TCP sequence number scatter plot                                     |
| `--tcp-error`               | Extract TCP analysis flags and plot error counts                               |
| `--formats {png,csv,svg,pdf} [ … ]` | Output formats (default: `png`)                                    |
| `--output-dir PATH`         | Directory for output files (defaults to same folder as each PCAP)             |
| `-h, --help`                | Show help message and exit                                                    |

---

## Examples

1. **Basic Throughput Plot (1 s bins)**  
   ```bash
   ./pcap_analyzer.py capture.pcap
   ```

2. **Specify Ports & Total**  
   ```bash
   ./pcap_analyzer.py capture.pcap \
     --port 80 443 \
     --port-legend "HTTP" "HTTPS" \
     --total
   ```

3. **TCP‐Only with Sequence & Error Bars**  
   ```bash
   ./pcap_analyzer.py capture.pcap \
     --tcp \
     --port 5000 \
     --sequence \
     --tcp-error \
     --formats png csv
   ```

4. **Multiple PCAPs in Parallel**  
   ```bash
   ./pcap_analyzer.py session1.pcap session2.pcap \
     --interval 0.5 \
     --output-dir analysis_results
   ```

---

## Output Files

For each input `foo.pcap` and selected formats, you’ll get:

```
foo.pcap.data_rate.png  # Throughput plot
foo.pcap.data_rate.csv  # Throughput CSV data
foo.pcap.data_rate.svg  # Vector throughput plot
foo.pcap.data_rate.pdf  # Throughput PDF

foo.pcap.seq.png        # TCP sequence scatter
foo.pcap.seq.csv        # Sequence CSV data
foo.pcap.seq.svg
foo.pcap.seq.pdf
```

---

## Customization

- **Time Bin Width**: adjust `--interval` to zoom in/out on throughput timescales  
- **Port Filtering**: use `--port` & `--port-legend` to focus on application ports  
- **Protocol Filtering**: `--tcp` or `--quic` for protocol‑specific analysis  
- **Output Formats**: mix & match `png`, `csv`, `svg`, `pdf`  

---
