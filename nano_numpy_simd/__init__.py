"""Python-facing API for the nano-numpy-simd tutorial package."""

from __future__ import annotations

from .pure_python import add as add_py
from .pure_python import div as div_py
from .pure_python import mul as mul_py
from .pure_python import sub as sub_py

try:
    from ._native import add, add_rust, div, div_rust, mul, mul_rust, sub, sub_rust
except ImportError as exc:  # pragma: no cover - exercised before maturin develop.
    raise ImportError(
        "The Rust extension is not built yet. Run `maturin develop` from the "
        "repository root, then import nano_numpy_simd again."
    ) from exc

__all__ = [
    "add",
    "add_py",
    "add_rust",
    "div",
    "div_py",
    "div_rust",
    "mul",
    "mul_py",
    "mul_rust",
    "sub",
    "sub_py",
    "sub_rust",
]
