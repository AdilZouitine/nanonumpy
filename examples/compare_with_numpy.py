"""Compare nano-numpy-simd results with NumPy."""

from __future__ import annotations

import numpy as np

import nano_numpy_simd as nn


def main() -> None:
    """Print NumPy and Rust SIMD results for the same inputs."""
    a = [1.0, 2.0, 3.0, 4.0]
    b = [10.0, 20.0, 30.0, 40.0]

    print("nano_numpy_simd:", nn.add(a, b))
    print("numpy:", (np.array(a, dtype=np.float32) + np.array(b, dtype=np.float32)).tolist())


if __name__ == "__main__":
    main()
