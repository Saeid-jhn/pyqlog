"""Unit auto-scaling helper.

Consolidates the three near-identical scaling routines that previously lived in
the qlog, iperf and pcap plotters.
"""

from __future__ import annotations

from typing import Sequence, Tuple

# Common ladders, largest threshold first (the order ``auto_scale`` expects).
BYTES = [(1e9, "GB"), (1e6, "MB"), (1e3, "KB"), (1.0, "B")]
BITS_PER_SEC = [(1e9, "Gb/s"), (1e6, "Mb/s"), (1e3, "Kb/s"), (1.0, "b/s")]


def auto_scale(
    value: float,
    units: Sequence[Tuple[float, str]] = BYTES,
) -> Tuple[float, str]:
    """Scale ``value`` by the first threshold it meets and return ``(scaled, unit)``.

    ``units`` must be ordered largest-threshold-first. If ``value`` is below every
    threshold, the smallest unit is used.
    """
    for thresh, unit in units:
        if value >= thresh:
            return value / thresh, unit
    return value, units[-1][1]


def scale_factor(
    value: float,
    units: Sequence[Tuple[float, str]] = BITS_PER_SEC,
) -> Tuple[float, str]:
    """Return ``(divisor, unit)`` for scaling a whole series by a single factor.

    Unlike :func:`auto_scale` this returns the raw divisor (e.g. ``1e6``) so a
    caller can divide an entire column/array and label the axis once.
    """
    for thresh, unit in units:
        if value >= thresh:
            return thresh, unit
    return 1.0, units[-1][1]
