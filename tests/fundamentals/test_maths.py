import numpy as np
import pytest

from fundamentals.maths import cosine_similarity, dot, matmul, norm, softmax


def test_dot_multiplies_pairs_and_sums():
    assert dot([1, 2, 3], [4, 5, 6]) == 32


def test_dot_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        dot([1, 2], [1, 2, 3])


def test_norm_is_vector_length():
    assert norm([3, 4]) == 5


@pytest.mark.parametrize(
    "a, b, expected",
    [
        ([1, 0], [2, 0], 1.0),   # same direction, different length
        ([1, 0], [-3, 0], -1.0),  # opposite direction
        ([1, 0], [0, 5], 0.0),   # at right angles
    ],
)
def test_cosine_similarity_measures_direction_not_length(a, b, expected):
    assert cosine_similarity(a, b) == pytest.approx(expected)


def test_cosine_similarity_rejects_zero_vector():
    with pytest.raises(ValueError):
        cosine_similarity([0, 0], [1, 2])


def test_matmul_matches_numpy():
    A = [[1, 2, 3], [4, 5, 6]]
    B = [[7, 8], [9, 10], [11, 12]]
    assert matmul(A, B) == (np.array(A) @ np.array(B)).tolist()


def test_matmul_rejects_incompatible_shapes():
    with pytest.raises(ValueError):
        matmul([[1, 2]], [[1, 2]])


def test_softmax_is_a_probability_distribution():
    p = softmax([2.0, 1.0, 0.1])
    assert p.sum() == pytest.approx(1.0)
    assert (p > 0).all()
    assert p.argmax() == 0


def test_softmax_survives_huge_logits():
    p = softmax([1000.0, 1001.0])
    assert not np.isnan(p).any()
    assert p.sum() == pytest.approx(1.0)


def test_low_temperature_sharpens_high_temperature_flattens():
    logits = [2.0, 1.0, 0.1]
    assert softmax(logits, temperature=0.1)[0] > 0.99
    assert softmax(logits, temperature=100.0).max() < 0.35


def test_softmax_rejects_non_positive_temperature():
    with pytest.raises(ValueError):
        softmax([1.0, 2.0], temperature=0)
