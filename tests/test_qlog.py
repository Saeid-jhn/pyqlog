"""Tests for the qlog/sqlog analyzer + facade."""

from pathlib import Path

import pyqlog
from pyqlog.qlog import QlogAnalyzer


def test_picoquic_analyze(qlog_file):
    result = QlogAnalyzer(qlog_file).analyze()
    assert result.kind == "qlog"
    assert set(result.tables) == {"packets", "metrics", "offsets",
                                   "datagram", "data_rate"}
    assert len(result.table("packets")) == 2
    # picoquic times are already microseconds -> unchanged
    assert result.table("packets")["time"].iloc[0] == 1000


def test_quiche_time_normalization(sqlog_file):
    """quiche ms timestamps and RTTs must be scaled to microseconds."""
    result = QlogAnalyzer(sqlog_file).analyze()
    packets = result.table("packets")
    assert packets["time"].iloc[0] == 1000  # 1 ms -> 1000 µs

    metrics = result.table("metrics")
    smoothed = metrics[metrics["key"] == "smoothed_rtt"]["value"].iloc[0]
    assert smoothed == 5000  # 5 ms -> 5000 µs
    # pacing_rate: bytes/s * 8 -> bits/s
    pacing = metrics[metrics["key"] == "pacing_rate"]["value"].iloc[0]
    assert pacing == 125000 * 8


def test_data_rate_computed(sqlog_file):
    result = QlogAnalyzer(sqlog_file, interval="1000ms").analyze()
    rate = result.table("data_rate")
    assert not rate.empty
    assert list(rate.columns) == ["start_interval (s)", "end_interval (s)",
                                  "throughput (bps)", "goodput (bps)"]


def test_plot_and_replot_roundtrip(sqlog_file, tmp_path):
    out = tmp_path / "out"
    written = pyqlog.plot(sqlog_file, out_dir=str(out), formats=["png"])
    assert written and all(Path(p).exists() for p in written)

    # CSVs were persisted alongside the plot
    prefix = out / "sample.sqlog"
    assert (out / "sample.sqlog.metrics.csv").exists()

    # replot from CSVs only, no original log
    replotted = pyqlog.replot(str(prefix), formats=["pdf"])
    assert replotted and all(Path(p).exists() for p in replotted)


def test_autodetect_kind(qlog_file, sqlog_file):
    assert pyqlog.detect_kind(qlog_file) == "qlog"
    assert pyqlog.detect_kind(sqlog_file) == "qlog"
