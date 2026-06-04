"""Parallel batch helper shared by the CLI subcommands."""

from __future__ import annotations

import logging
from multiprocessing import Pool, cpu_count
from typing import Callable, Iterable, List, Sequence, Tuple, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


def run_many(
    fn: Callable[..., R],
    items: Sequence[T],
    *,
    extra_args: Tuple = (),
    parallel: bool = True,
) -> List[R]:
    """Apply ``fn(item, *extra_args)`` to every item, optionally in parallel.

    Falls back to sequential execution for a single item or when ``parallel`` is
    False (handy for debugging, since multiprocessing hides tracebacks).
    """
    work: Iterable[Tuple] = [(item, *extra_args) for item in items]

    if not parallel or len(items) <= 1:
        return [fn(*args) for args in work]

    with Pool(min(cpu_count(), len(items))) as pool:
        return pool.starmap(fn, work)
