"""Tests for the zero-copy buffer protocol API."""

from __future__ import annotations

import numpy as np
import pytest

import nano_numpy_simd as nn


def test_add_into_writes_to_preallocated_numpy_output() -> None:
    """The buffer API writes into an existing float32 NumPy array."""
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    b = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    out = np.empty_like(a)

    result = nn.add_into(a, b, out)

    assert result is None
    np.testing.assert_allclose(out, np.array([11.0, 22.0, 33.0, 44.0], dtype=np.float32))


def test_buffer_operations_match_numpy() -> None:
    """All buffer operations match NumPy for tail lengths."""
    rng = np.random.default_rng(123)
    a = rng.random(17, dtype=np.float32) + np.float32(1.0)
    b = rng.random(17, dtype=np.float32) + np.float32(1.0)

    operations = [
        (nn.add_into, np.add),
        (nn.sub_into, np.subtract),
        (nn.mul_into, np.multiply),
        (nn.div_into, np.divide),
    ]

    for func, np_func in operations:
        out = np.empty_like(a)
        func(a, b, out)
        np.testing.assert_allclose(out, np_func(a, b), rtol=1e-5, atol=1e-6)


def test_add_into_requires_writable_output() -> None:
    """The output buffer must be writable."""
    a = np.array([1.0], dtype=np.float32)
    b = np.array([2.0], dtype=np.float32)
    out = np.empty_like(a)
    out.flags.writeable = False

    with pytest.raises(ValueError, match="output buffer must be writable"):
        nn.add_into(a, b, out)


def test_add_into_rejects_output_aliasing_input() -> None:
    """Aliasing output with input would break Rust slice aliasing rules."""
    a = np.array([1.0], dtype=np.float32)
    b = np.array([2.0], dtype=np.float32)

    with pytest.raises(ValueError, match="output buffer must not overlap"):
        nn.add_into(a, b, a)
