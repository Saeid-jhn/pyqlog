# quiclog (qlog)

## Description

This project provides tools for parsing QUIC protocol qlog files for the following QUIC implementations:

- picoquic (`.qlog`)
- quiche (`.sqlog`)

## Requirements
- Python 3.6 or higher
- Pandas
- Matplotlib
- Seaborn

## Usage
This tool processes QUIC log files, either a single file or all files in parallel in a specified directory, and generates visualizations based on the extracted data. The script expects files to be in a specific format, namely `filename.[QUIC logging format]`.

### Processing All Files in a Directory
To process all qlog files in a directory, run the script with the directory path as an argument:
```
python path/to/script.py /path/to/qlog/directory/
```

### Processing a Specific File
If you want to process a specific qlog file, use the `--file` option followed by the filename:
```
python path/to/script.py /path/to/qlog/directory --file example.qlog
```
