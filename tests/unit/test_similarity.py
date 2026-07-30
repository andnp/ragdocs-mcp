import pytest

from searchkernel.utils.similarity import cosine_similarity_lists


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ([1.0, 0.0], [1.0, 0.0], 1.0),
        ([1.0, 0.0], [0.0, 1.0], 0.0),
        ([0.0, 0.0], [1.0, 0.0], 0.0),
        ([1.0], [1.0, 0.0], 0.0),
    ],
)
def test_cosine_similarity_lists_handles_common_vector_shapes(
    left: list[float],
    right: list[float],
    expected: float,
) -> None:
    assert cosine_similarity_lists(left, right) == expected
