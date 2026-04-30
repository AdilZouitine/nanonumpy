"""Basic API correctness tests."""

from __future__ import annotations

import nano_numpy_simd as nn


def test_add_versions_match() -> None:
    """All add implementations produce the same values."""
    a = [1.0, 2.0, 3.0, 4.0]
    b = [10.0, 20.0, 30.0, 40.0]

    assert nn.add_py(a, b) == [11.0, 22.0, 33.0, 44.0]
    assert nn.add_rust(a, b) == nn.add_py(a, b)
    assert nn.add(a, b) == nn.add_py(a, b)


def test_all_operations() -> None:
    """The four arithmetic operations work for a small input."""
    a = [8.0, 6.0]
    b = [2.0, 3.0]

    assert nn.add(a, b) == [10.0, 9.0]
    assert nn.sub(a, b) == [6.0, 3.0]
    assert nn.mul(a, b) == [16.0, 18.0]
    assert nn.div(a, b) == [4.0, 2.0]


def test_empty_arrays() -> None:
    """Empty arrays return empty arrays."""
    assert nn.add([], []) == []
    assert nn.add_rust([], []) == []
    assert nn.add_py([], []) == []
