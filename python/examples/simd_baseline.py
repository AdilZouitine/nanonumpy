"""Run the SIMD-dispatched Rust implementation."""

from __future__ import annotations

import nano_numpy_simd as nn


def main() -> None:
    """Print the SIMD-dispatched result."""
    print(nn.add([1.0, 2.0], [10.0, 20.0]))


if __name__ == "__main__":
    main()
