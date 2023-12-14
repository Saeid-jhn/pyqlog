# pyqlog

## Description

This project provides tools for parsing QUIC protocol qlog files for the following QUIC implementations:

- picoquic (`.qlog`)
- quiche (`.sqlog`)

## Requirements
- Python 3.9 or higher
- Pandas
- Matplotlib
- Seaborn

## Usage
This tool processes single or multple QUIC log files and generates visualizations based on the extracted data. The script expects files to be in a specific format, namely `filename.[QUIC logging format]`.

```
qlog_plot.py [-h] [--debug] file [file ...]
```

To enable **DEBUG** mode, use the `--debug` option.
