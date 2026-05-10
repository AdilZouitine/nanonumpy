"""Run the pure Python baseline."""

from __future__ import annotations

import nano_numpy_simd as nn


def main() -> None:
    """Print the pure Python result."""
    print(nn.add_py([1.0, 2.0], [10.0, 20.0]))


if __name__ == "__main__":
    main()
