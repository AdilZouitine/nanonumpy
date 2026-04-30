"""Error handling tests."""

from __future__ import annotations

import pytest

import nano_numpy_simd as nn


def test_length_mismatch_raises_value_error() -> None:
    """Every public implementation validates matching lengths."""
    functions = [nn.add_py, nn.add_rust, nn.add, nn.sub_py, nn.sub_rust, nn.sub]

    for func in functions:
        with pytest.raises(ValueError, match="arrays must have the same length"):
            func([1.0], [1.0, 2.0])
