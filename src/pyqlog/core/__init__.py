"""Shared building blocks used by every pyqlog analyzer and plotter."""

from .result import AnalysisResult
from .units import auto_scale
from .plotting import apply_style, save_figure, PNG_DPI
from .batch import run_many

__all__ = [
    "AnalysisResult",
    "auto_scale",
    "apply_style",
    "save_figure",
    "PNG_DPI",
    "run_many",
]
