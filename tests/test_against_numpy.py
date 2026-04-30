"""Correctness tests against NumPy."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

import nano_numpy_simd as nn


ArrayOp = Callable[[list[float], list[float]], list[float]]
NumpyOp = Callable[[np.ndarray, np.ndarray], np.ndarray]


def _check_operation(op: ArrayOp, np_op: NumpyOp, length: int) -> None:
    """Compare one nano-numpy-simd operation against NumPy."""
    rng = np.random.default_rng(length)
    a_np = rng.random(length, dtype=np.float32) + np.float32(1.0)
    b_np = rng.random(length, dtype=np.float32) + np.float32(1.0)
    expected = np_op(a_np, b_np)

    result = np.array(op(a_np.tolist(), b_np.tolist()), dtype=np.float32)

    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)


def test_operations_match_numpy_for_tail_lengths() -> None:
    """Lengths around vector widths exercise scalar tail handling."""
    lengths = [0, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17]
    operations: list[tuple[ArrayOp, NumpyOp]] = [
        (nn.add, np.add),
        (nn.sub, np.subtract),
        (nn.mul, np.multiply),
        (nn.div, np.divide),
    ]

    for length in lengths:
        for op, np_op in operations:
            _check_operation(op, np_op, length)
