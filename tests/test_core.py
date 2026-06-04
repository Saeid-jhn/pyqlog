"""Tests for shared core utilities and the AnalysisResult contract."""

import pandas as pd
import pytest

from pyqlog.core import AnalysisResult, auto_scale
from pyqlog.core.units import BYTES, BITS_PER_SEC


def test_auto_scale():
    assert auto_scale(2_500_000, BYTES) == (2.5, "MB")
    assert auto_scale(1500, BYTES) == (1.5, "KB")
    assert auto_scale(0.5, BYTES) == (0.5, "B")
    assert auto_scale(3e9, BITS_PER_SEC) == (3.0, "Gb/s")


def test_result_csv_roundtrip(tmp_path):
    res = AnalysisResult(
        kind="qlog",
        tables={"metrics": pd.DataFrame({"key": ["rtt"], "value": [5]})},
        metadata={"interval": "1000ms"},
        source="x.qlog",
    )
    prefix = tmp_path / "run"
    res.to_csv(str(prefix))

    loaded = AnalysisResult.from_csv(str(prefix))
    assert loaded.kind == "qlog"
    assert loaded.metadata["interval"] == "1000ms"
    assert loaded.table("metrics")["value"].iloc[0] == 5


def test_from_csv_requires_kind(tmp_path):
    df = pd.DataFrame({"a": [1]})
    (tmp_path / "bare.t.csv").write_text(df.to_csv(index=False))
    with pytest.raises(ValueError):
        AnalysisResult.from_csv(str(tmp_path / "bare"))


def test_detect_kind_unknown():
    from pyqlog.registry import detect_kind
    with pytest.raises(ValueError):
        detect_kind("mystery.bin")
