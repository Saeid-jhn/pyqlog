"""Parallel batch helper shared by the CLI subcommands."""

from __future__ import annotations

import logging
from multiprocessing import Pool, cpu_count
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


def run_many(
    fn: Callable[..., R],
    items: Sequence[T],
    *,
    extra_args: Tuple = (),
    parallel: bool = True,
    workers: Optional[int] = None,
) -> List[R]:
    """Apply ``fn(item, *extra_args)`` to every item, optionally in parallel.

    ``workers`` caps the process-pool size (default: one per CPU). Falls back to
    sequential execution for a single item, when ``parallel`` is False, or when
    the effective pool size is 1 (handy for debugging, since multiprocessing
    hides tracebacks).
    """
    work: Iterable[Tuple] = [(item, *extra_args) for item in items]

    n = min(workers or cpu_count(), len(items))
    if not parallel or n <= 1:
        return [fn(*args) for args in work]

    with Pool(n) as pool:
        return pool.starmap(fn, work)
