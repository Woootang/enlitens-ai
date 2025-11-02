"""Utility helpers for vector math with optional NumPy support."""
from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Union

try:  # Optional dependency - allow repo to run without numpy installed
    import numpy as _np  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - exercised in CI without numpy
    _np = None

VectorLike = Union[Sequence[float], "_np.ndarray"]  # type: ignore[name-defined]
MatrixLike = Union[Sequence[Sequence[float]], "_np.ndarray"]  # type: ignore[name-defined]


def numpy_available() -> bool:
    """Return ``True`` when NumPy is available in the environment."""

    return _np is not None


def ensure_float_list(vector: VectorLike) -> List[float]:
    """Convert a vector into a plain Python ``list`` of ``float`` values."""

    if numpy_available() and isinstance(vector, _np.ndarray):  # type: ignore[arg-type]
        return vector.astype("float32").tolist()
    return [float(value) for value in vector]


def ensure_2d_float_list(vectors: MatrixLike) -> List[List[float]]:
    """Convert a matrix-like structure into a list of float vectors."""

    if numpy_available() and isinstance(vectors, _np.ndarray):  # type: ignore[arg-type]
        return [row.astype("float32").tolist() for row in vectors]  # pragma: no cover - simple branch
    return [ensure_float_list(row) for row in vectors]


def normalize(vector: VectorLike) -> List[float]:
    """Return the L2-normalised version of ``vector`` as a list of floats."""

    values = ensure_float_list(vector)
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0.0:
        return values
    return [value / norm for value in values]


def dot(vector_a: VectorLike, vector_b: VectorLike) -> float:
    """Compute the dot product between two vectors."""

    a = ensure_float_list(vector_a)
    b = ensure_float_list(vector_b)
    if len(a) != len(b):  # pragma: no cover - defensive guard
        raise ValueError("Vectors must be the same length")
    return float(sum(x * y for x, y in zip(a, b)))


def mean(vectors: Iterable[VectorLike]) -> List[float]:
    """Compute the element-wise mean of the provided vectors."""

    sequence = [ensure_float_list(vector) for vector in vectors]
    if not sequence:
        raise ValueError("At least one vector is required to compute the mean")

    length = len(sequence[0])
    totals = [0.0] * length
    for vector in sequence:
        if len(vector) != length:  # pragma: no cover - defensive guard
            raise ValueError("Vectors must have identical dimensions")
        for index, value in enumerate(vector):
            totals[index] += value
    count = float(len(sequence))
    return [value / count for value in totals]


def to_numpy(vector: Sequence[float]):
    """Convert a float sequence into a ``numpy.ndarray`` when available."""

    if not numpy_available():
        raise RuntimeError("NumPy is not available in this environment")
    return _np.asarray(list(vector), dtype=_np.float32)  # type: ignore[attr-defined]


def to_numpy_matrix(vectors: Sequence[Sequence[float]]):
    """Convert a matrix into a ``numpy.ndarray`` when available."""

    if not numpy_available():
        raise RuntimeError("NumPy is not available in this environment")
    return _np.asarray([list(vector) for vector in vectors], dtype=_np.float32)  # type: ignore[attr-defined]
