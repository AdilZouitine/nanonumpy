"""Pure Python arithmetic baselines for the tutorial."""

from __future__ import annotations

from collections.abc import Sequence


NumberSequence = Sequence[float]


def _validate_lengths(a: NumberSequence, b: NumberSequence) -> None:
    """Raise ValueError when two arrays do not have the same length.

    Args:
        a: First one-dimensional input sequence.
        b: Second one-dimensional input sequence.

    Raises:
        ValueError: If the two inputs have different lengths.
    """
    if len(a) != len(b):
        raise ValueError("arrays must have the same length")


def add(a: NumberSequence, b: NumberSequence) -> list[float]:
    """Add two one-dimensional arrays with a Python list comprehension.

    Args:
        a: First one-dimensional input sequence.
        b: Second one-dimensional input sequence.

    Returns:
        A new list containing `a[i] + b[i]`.
    """
    _validate_lengths(a, b)
    return [x + y for x, y in zip(a, b)]


def sub(a: NumberSequence, b: NumberSequence) -> list[float]:
    """Subtract two one-dimensional arrays with a Python list comprehension.

    Args:
        a: First one-dimensional input sequence.
        b: Second one-dimensional input sequence.

    Returns:
        A new list containing `a[i] - b[i]`.
    """
    _validate_lengths(a, b)
    return [x - y for x, y in zip(a, b)]


def mul(a: NumberSequence, b: NumberSequence) -> list[float]:
    """Multiply two one-dimensional arrays with a Python list comprehension.

    Args:
        a: First one-dimensional input sequence.
        b: Second one-dimensional input sequence.

    Returns:
        A new list containing `a[i] * b[i]`.
    """
    _validate_lengths(a, b)
    return [x * y for x, y in zip(a, b)]


def div(a: NumberSequence, b: NumberSequence) -> list[float]:
    """Divide two one-dimensional arrays with a Python list comprehension.

    Args:
        a: First one-dimensional input sequence.
        b: Second one-dimensional input sequence.

    Returns:
        A new list containing `a[i] / b[i]`.
    """
    _validate_lengths(a, b)
    return [x / y for x, y in zip(a, b)]
