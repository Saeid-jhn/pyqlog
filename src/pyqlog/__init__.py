"""pyqlog -- parse & visualize QUIC qlog, pcap and iperf3 network logs.

Quick start::

    import pyqlog
    pyqlog.plot("run1.sqlog", out_dir="results")           # analyze + plot
    result = pyqlog.analyze("run1.sqlog")                   # just the data
    pyqlog.replot("results/run1.sqlog", formats=["pdf"])    # re-plot, no rerun

For full control, use the explicit classes (``QlogAnalyzer``/``QlogPlotter``,
``IperfAnalyzer``/``IperfPlotter``, ``PcapAnalyzer``/``PcapPlotter``).
"""

from .core import AnalysisResult
from .facade import analyze, plot, plot_many, replot
from .registry import detect_kind
from .qlog import QlogAnalyzer, QlogPlotter
from .iperf import IperfAnalyzer, IperfPlotter
from .pcap import PcapAnalyzer, PcapPlotter

__version__ = "0.3.0"

__all__ = [
    "analyze",
    "plot",
    "plot_many",
    "replot",
    "detect_kind",
    "AnalysisResult",
    "QlogAnalyzer",
    "QlogPlotter",
    "IperfAnalyzer",
    "IperfPlotter",
    "PcapAnalyzer",
    "PcapPlotter",
    "__version__",
]
