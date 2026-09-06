"""Golden-value tests for the in-tree contrast matrix construction.

These values were captured from ``patsy.contrasts`` so the in-tree
implementation can be diffed against the previous patsy behaviour
without keeping patsy installed.
"""

import numpy as np

from category_encoders._contrasts import ContrastMatrix, Diff, Helmert, Poly, Sum


def _check(actual: ContrastMatrix, expected_matrix, expected_suffixes):
    np.testing.assert_allclose(actual.matrix, expected_matrix, atol=1e-12)
    assert actual.column_suffixes == expected_suffixes


def test_sum_n4():
    _check(
        Sum().code_without_intercept(range(4)),
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, -1, -1]],
        ['[S.0]', '[S.1]', '[S.2]'],
    )


def test_helmert_n4():
    _check(
        Helmert().code_without_intercept(range(4)),
        [[-1, -1, -1], [1, -1, -1], [0, 2, -1], [0, 0, 3]],
        ['[H.1]', '[H.2]', '[H.3]'],
    )


def test_diff_n4():
    _check(
        Diff().code_without_intercept(range(4)),
        [
            [-0.75, -0.5, -0.25],
            [0.25, -0.5, -0.25],
            [0.25, 0.5, -0.25],
            [0.25, 0.5, 0.75],
        ],
        ['[D.0]', '[D.1]', '[D.2]'],
    )


def test_poly_n4():
    sqrt5 = np.sqrt(5)
    _check(
        Poly().code_without_intercept(range(4)),
        [
            [-1.5 / sqrt5, 0.5, -0.5 / sqrt5],
            [-0.5 / sqrt5, -0.5, 1.5 / sqrt5],
            [0.5 / sqrt5, -0.5, -1.5 / sqrt5],
            [1.5 / sqrt5, 0.5, 0.5 / sqrt5],
        ],
        ['.Linear', '.Quadratic', '.Cubic'],
    )


def test_poly_columns_are_orthonormal():
    q = Poly().code_without_intercept(range(7)).matrix
    np.testing.assert_allclose(q.T @ q, np.eye(6), atol=1e-12)


def test_all_contrasts_have_zero_column_means():
    for cls in (Sum, Helmert, Diff, Poly):
        m = cls().code_without_intercept(range(5)).matrix
        np.testing.assert_allclose(m.mean(axis=0), np.zeros(m.shape[1]), atol=1e-12)


def test_n_equals_one_returns_empty_matrix():
    for cls in (Sum, Helmert, Diff, Poly):
        cm = cls().code_without_intercept([0])
        assert cm.matrix.shape == (1, 0)
        assert cm.column_suffixes == []


def test_poly_high_degree_suffixes_use_caret():
    suffixes = Poly().code_without_intercept(range(6)).column_suffixes
    assert suffixes == ['.Linear', '.Quadratic', '.Cubic', '^4', '^5']
