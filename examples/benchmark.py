"""Small benchmark for the three tutorial levels."""

from __future__ import annotations

from collections.abc import Callable
from time import perf_counter

import nano_numpy_simd as nn


ArrayOp = Callable[[list[float], list[float]], list[float]]


def time_once(func: ArrayOp, a: list[float], b: list[float]) -> float:
    """Return elapsed seconds for one function call."""
    start = perf_counter()
    func(a, b)
    return perf_counter() - start


def main() -> None:
    """Benchmark pure Python, naive Rust, and SIMD-dispatched Rust."""
    size = 100_000
    a = [float(i) for i in range(size)]
    b = [float(i + 1) for i in range(size)]

    for name, func in [("Pure Python", nn.add_py), ("Naive Rust", nn.add_rust), ("Rust SIMD", nn.add)]:
        elapsed = time_once(func, a, b)
        print(f"{name:12} {elapsed * 1_000:8.3f} ms")


if __name__ == "__main__":
    main()
