"""Tests for the iperf3 analyzer + facade."""

from pathlib import Path

import pyqlog
from pyqlog.iperf import IperfAnalyzer, INTERVAL_COLS


def test_iperf_analyze(iperf_file):
    result = IperfAnalyzer(iperf_file).analyze()
    assert result.kind == "iperf"
    df = result.table("intervals")
    assert list(df.columns) == INTERVAL_COLS
    assert len(df) == 2
    assert df["snd_throughput (bps)"].iloc[0] == 1_000_000
    # metadata captured from the start event
    assert result.metadata["remote_host"] == "10.0.0.2"
    assert result.metadata["test_start.protocol"] == "TCP"


def test_iperf_plot_and_replot(iperf_file, tmp_path):
    out = tmp_path / "out"
    written = pyqlog.plot(iperf_file, out_dir=str(out), formats=["png"], title=True)
    assert written and all(Path(p).exists() for p in written)

    prefix = out / "sample.json"
    assert (out / "sample.json.intervals.csv").exists()
    assert (out / "sample.json.meta.json").exists()

    replotted = pyqlog.replot(str(prefix), formats=["png"])
    assert replotted and all(Path(p).exists() for p in replotted)


def test_iperf_metric_subset(iperf_file, tmp_path):
    result = pyqlog.analyze(iperf_file)
    from pyqlog.iperf import IperfPlotter
    written = IperfPlotter(result, metrics=["rtt"]).render(
        out_dir=str(tmp_path), formats=["png"])
    assert written
