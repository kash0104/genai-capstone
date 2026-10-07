import math

import numpy as np


def dot(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
    total = 0
    for x, y in zip(a, b):
        total = total + x * y
    return total


def norm(a: list[float]) -> float:
    return math.sqrt(dot(a, a))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    length_a = norm(a)
    length_b = norm(b)
    if length_a == 0 or length_b == 0:
        raise ValueError("cosine similarity is undefined for a zero vector")
    return dot(a,b) / (length_a * length_b)


def matmul(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    if len(A[0]) != len(B):
        raise ValueError(f"shape mismatch: {len(A)}x{len(A[0])} @ {len(B)}x{len(B[0])}")
    columns = list(zip(*B))
    return [[dot(row, col) for col in columns] for row in A]


def softmax(logits: list[float], temperature: float = 1.0) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be > 0; APIs treat 0 as a special 'always pick the top token' case")
    z = np.asarray(logits, dtype=float) / temperature
    z = z - z.max()  # shifting every score equally leaves the result unchanged but keeps exp() from overflowing
    e = np.exp(z)
    return e / e.sum()