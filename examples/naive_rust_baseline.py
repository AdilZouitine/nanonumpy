"""Run the naive Rust baseline through PyO3."""

from __future__ import annotations

import nano_numpy_simd as nn


def main() -> None:
    """Print the naive Rust result."""
    print(nn.add_rust([1.0, 2.0], [10.0, 20.0]))


if __name__ == "__main__":
    main()
