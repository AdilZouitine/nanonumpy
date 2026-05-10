"""Benchmark the preallocated buffer API against the list API and NumPy."""

from __future__ import annotations

from collections.abc import Callable
from time import perf_counter

import numpy as np

import nano_numpy_simd as nn


def best_time(func: Callable[[], object], repeats: int = 10) -> float:
    """Return the best elapsed time across repeated calls."""
    best = float("inf")
    for _ in range(repeats):
        start = perf_counter()
        func()
        best = min(best, perf_counter() - start)
    return best


def main() -> None:
    """Compare list conversion, preallocated buffers, and NumPy."""
    size = 5_000_000
    # The buffer API needs contiguous float32 memory. NumPy arrays are a good
    # teaching input because they expose the Python buffer protocol.
    a_np = np.arange(size, dtype=np.float32)
    b_np = np.arange(size, dtype=np.float32) + np.float32(1.0)
    # `out` is allocated once and reused. This avoids timing output allocation
    # inside `nn.add_into`.
    out = np.empty_like(a_np)
    # Convert once so the list API benchmark measures the call path, not setup.
    a_list = a_np.tolist()
    b_list = b_np.tolist()

    timings = [
        ("Pure Python list", best_time(lambda: nn.add_py(a_list, b_list), repeats=3)),
        ("Rust list API", best_time(lambda: nn.add(a_list, b_list), repeats=3)),
        ("Rust buffer into", best_time(lambda: nn.add_into(a_np, b_np, out))),
        ("NumPy allocated", best_time(lambda: np.add(a_np, b_np))),
        ("NumPy out=", best_time(lambda: np.add(a_np, b_np, out=out))),
    ]
    baseline = timings[0][1]

    print(f"Size: {size:,} float32 values")
    print(f"{'Implementation':18} {'Time':14} {'Speedup vs Python'}")
    for name, seconds in timings:
        print(f"{name:18} {seconds * 1_000:10.3f} ms   {baseline / seconds:8.2f}x")


if __name__ == "__main__":
    main()
