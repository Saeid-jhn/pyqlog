"""Shared pytest fixtures: headless plotting + tiny synthetic log files."""

import json

import matplotlib
import pytest

matplotlib.use("Agg")  # no display needed in CI

RS = "\x1e"  # quiche JSON-SEQ record separator (ASCII 0x1E)


@pytest.fixture
def qlog_file(tmp_path):
    """A minimal picoquic .qlog (server role, two packets + metrics)."""
    doc = {
        "traces": [{
            "vantage_point": {"type": "server"},
            "events": [
                [1000, "transport", "packet_sent",
                 {"header": {"packet_number": 1, "packet_size": 1200},
                  "frames": [{"frame_type": "stream", "offset": 0, "length": 1000}]}],
                [1000, "recovery", "metrics_updated",
                 {"smoothed_rtt": 5000, "cwnd": 12000, "bytes_in_flight": 2400,
                  "pacing_rate": 1000000}],
                [2000000, "transport", "packet_sent",
                 {"header": {"packet_number": 2, "packet_size": 1200},
                  "frames": [{"frame_type": "stream", "offset": 1000, "length": 1000}]}],
            ],
        }],
    }
    p = tmp_path / "sample.qlog"
    p.write_text(json.dumps(doc))
    return str(p)


@pytest.fixture
def sqlog_file(tmp_path):
    """A minimal quiche .sqlog (server role; times in ms)."""
    header = {"title": "server"}
    e1 = {"time": 1, "name": "transport:packet_sent",
          "data": {"header": {"packet_number": 1},
                   "raw": {"length": 1200},
                   "frames": [{"frame_type": "stream", "offset": 0, "length": 1000}]}}
    e2 = {"time": 1, "name": "recovery:metrics_updated",
          "data": {"smoothed_rtt": 5, "min_rtt": 4, "latest_rtt": 6,
                   "cwnd": 12000, "pacing_rate": 125000}}
    e3 = {"time": 2000, "name": "transport:packet_sent",
          "data": {"header": {"packet_number": 2},
                   "raw": {"length": 1200},
                   "frames": [{"frame_type": "stream", "offset": 1000, "length": 1000}]}}
    body = RS.join(json.dumps(o) for o in (header, e1, e2, e3))
    p = tmp_path / "sample.sqlog"
    p.write_text(body)
    return str(p)


@pytest.fixture
def iperf_file(tmp_path):
    """A minimal iperf3 JSON-L log: one start event + two 1s intervals."""
    start = {"event": "start", "data": {
        "connected": [{"local_host": "10.0.0.1", "remote_host": "10.0.0.2",
                       "remote_port": 5201}],
        "version": "iperf 3.9",
        "timestamp": {"time": "Mon, 01 Jan 2024 10:00:00 GMT"},
        "tcp_mss_default": 1448, "sndbuf_actual": 16384, "rcvbuf_actual": 131072,
        "test_start": {"protocol": "TCP", "bytes": 1000000, "reverse": 0}}}

    def interval(beg, end):
        return {"event": "interval", "data": {"streams": [{
            "start": beg, "end": end, "sender": True, "bits_per_second": 1_000_000,
            "retransmits": 0, "snd_cwnd": 14480, "snd_wnd": 65535,
            "rtt": 5000, "rttvar": 1000}]}}

    lines = [start, interval(0, 1), interval(1, 2)]
    p = tmp_path / "sample.json"
    p.write_text("\n".join(json.dumps(o) for o in lines))
    return str(p)
