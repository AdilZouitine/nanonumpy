"""Cross-level correctness tests for benchmark paths."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

import nano_numpy_simd as nn


ArrayOp = Callable[[list[float], list[float]], list[float]]


def test_add_py_add_rust_add_and_numpy_match() -> None:
    """The benchmarked add implementations agree within float tolerance."""
    rng = np.random.default_rng(123)
    a_np = rng.random(1_024, dtype=np.float32) + np.float32(1.0)
    b_np = rng.random(1_024, dtype=np.float32) + np.float32(1.0)
    a = a_np.tolist()
    b = b_np.tolist()
    expected = np.add(a_np, b_np)

    for func in [nn.add_py, nn.add_rust, nn.add]:
        result = np.array(func(a, b), dtype=np.float32)
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)
