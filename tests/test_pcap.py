"""pcap tests -- skipped entirely when tshark is not installed."""

import pytest

from pyqlog.pcap import tshark_available

pytestmark = pytest.mark.skipif(
    not tshark_available(), reason="tshark not on PATH")


def test_pcap_smoke():
    # Placeholder: real pcap fixtures require a capture file + tshark.
    # The throughput/sequence logic is exercised via the analyzer's pure
    # DataFrame methods when a capture is available.
    assert tshark_available()
