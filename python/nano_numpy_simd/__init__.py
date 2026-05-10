"""Python-facing API for the nano-numpy-simd tutorial package."""

from __future__ import annotations

from .pure_python import add as add_py
from .pure_python import div as div_py
from .pure_python import mul as mul_py
from .pure_python import sub as sub_py

try:
    # `._native` is the shared library produced by maturin from the Rust crate.
    # After `uv run maturin develop --release`, Python can import it like a
    # normal Python module even though the implementation is compiled Rust.
    from ._native import (
        add,
        add_into,
        add_rust,
        div,
        div_into,
        div_rust,
        mul,
        mul_into,
        mul_rust,
        sub,
        sub_into,
        sub_rust,
    )
except ImportError as exc:  # pragma: no cover - exercised before maturin develop.
    raise ImportError(
        "The Rust extension is not built yet. Run "
        "`uv run maturin develop --release` from the repository root, then "
        "import nano_numpy_simd again."
    ) from exc

__all__ = [
    "add",
    "add_into",
    "add_py",
    "add_rust",
    "div",
    "div_into",
    "div_py",
    "div_rust",
    "mul",
    "mul_into",
    "mul_py",
    "mul_rust",
    "sub",
    "sub_into",
    "sub_py",
    "sub_rust",
]
