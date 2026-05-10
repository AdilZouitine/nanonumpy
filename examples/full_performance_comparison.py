"""Compare pure Python, naive Rust, SIMD Rust, and NumPy.

The printed numbers depend on your CPU, compiler, Python version, and system
load. They are measurements, not promises.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter

import numpy as np

import nano_numpy_simd as nn


ArrayOp = Callable[[list[float], list[float]], list[float]]
NumpyOp = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class TimedResult:
    """Elapsed time for one named implementation."""

    name: str
    seconds: float


def _time_python(func: ArrayOp, a: list[float], b: list[float]) -> TimedResult:
    """Time a list-based implementation."""
    start = perf_counter()
    func(a, b)
    return TimedResult(func.__name__, perf_counter() - start)


def _time_numpy(name: str, func: NumpyOp, a: np.ndarray, b: np.ndarray) -> TimedResult:
    """Time a NumPy implementation."""
    start = perf_counter()
    func(a, b)
    return TimedResult(name, perf_counter() - start)


def _assert_close(expected: np.ndarray, values: list[float]) -> None:
    """Verify that a Python-list result is close to NumPy."""
    np.testing.assert_allclose(np.array(values, dtype=np.float32), expected, rtol=1e-5, atol=1e-6)


def _benchmark_operation(
    name: str,
    py_op: ArrayOp,
    rust_op: ArrayOp,
    simd_op: ArrayOp,
    numpy_op: NumpyOp,
    size: int,
) -> None:
    """Benchmark one operation at one input size."""
    rng = np.random.default_rng(42)
    left_np = rng.random(size, dtype=np.float32) + np.float32(1.0)
    right_np = rng.random(size, dtype=np.float32) + np.float32(1.0)
    left = left_np.tolist()
    right = right_np.tolist()
    expected = numpy_op(left_np, right_np)

    _assert_close(expected, py_op(left, right))
    _assert_close(expected, rust_op(left, right))
    _assert_close(expected, simd_op(left, right))

    results = [
        TimedResult("Pure Python", _time_python(py_op, left, right).seconds),
        TimedResult("Naive Rust", _time_python(rust_op, left, right).seconds),
        TimedResult("Rust SIMD", _time_python(simd_op, left, right).seconds),
        _time_numpy("NumPy", numpy_op, left_np, right_np),
    ]
    baseline = results[0].seconds

    print(f"\nOperation: {name}")
    print(f"Size: {size:,}\n")
    print(f"{'Implementation':18} {'Time':14} {'Speedup vs Python'}")
    for result in results:
        speedup = baseline / result.seconds if result.seconds > 0 else float("inf")
        print(f"{result.name:18} {result.seconds * 1_000:10.3f} ms   {speedup:8.2f}x")


def main() -> None:
    """Run the full performance comparison."""
    operations: list[tuple[str, ArrayOp, ArrayOp, ArrayOp, NumpyOp]] = [
        ("add", nn.add_py, nn.add_rust, nn.add, np.add),
        ("sub", nn.sub_py, nn.sub_rust, nn.sub, np.subtract),
        ("mul", nn.mul_py, nn.mul_rust, nn.mul, np.multiply),
        ("div", nn.div_py, nn.div_rust, nn.div, np.divide),
    ]
    sizes = [1_000, 10_000, 100_000, 1_000_000, 10_000_000]

    for size in sizes:
        for name, py_op, rust_op, simd_op, numpy_op in operations:
            _benchmark_operation(name, py_op, rust_op, simd_op, numpy_op, size)

    print("\nNote: small arrays can be dominated by Python/Rust conversion overhead.")


if __name__ == "__main__":
    main()
