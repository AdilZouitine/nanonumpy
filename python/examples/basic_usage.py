"""Basic package usage."""

from __future__ import annotations

import nano_numpy_simd as nn


def main() -> None:
    """Run the smallest end-to-end example."""
    a = [1.0, 2.0, 3.0, 4.0]
    b = [10.0, 20.0, 30.0, 40.0]

    print(nn.add_py(a, b))
    print(nn.add_rust(a, b))
    print(nn.add(a, b))


if __name__ == "__main__":
    main()
