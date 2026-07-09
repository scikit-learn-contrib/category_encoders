"""Contrast matrix construction for the four contrast-coding encoders.

Replaces ``patsy.contrasts`` for the subset that this library uses
(``ContrastMatrix``, ``Poly``, ``Helmert``, ``Diff``, ``Sum``), keeping
the same call signature so encoder code does not need to know whether
the source is patsy or this module.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence

import numpy as np


class ContrastMatrix(NamedTuple):
    """A contrast matrix and the per-column suffixes that label it."""

    matrix: np.ndarray
    column_suffixes: list[str]


def _empty(n: int) -> ContrastMatrix:
    return ContrastMatrix(np.zeros((n, 0)), [])


class Poly:
    """Orthogonal polynomial contrasts on equally-spaced levels."""

    def code_without_intercept(self, levels: Sequence) -> ContrastMatrix:
        n = len(levels)
        if n < 2:
            return _empty(n)
        x = np.arange(n, dtype=float) - (n - 1) / 2
        # Include the constant column so Gram-Schmidt orthogonalizes the
        # polynomial columns against it (otherwise even-degree columns
        # retain a nonzero mean). Drop the intercept column afterward.
        basis = np.column_stack([np.ones(n), *[x**k for k in range(1, n)]])
        q, _ = np.linalg.qr(basis)
        q = q[:, 1:]
        # Match patsy's sign convention: at row 0, column for degree d
        # should have sign (-1)**d since x[0] < 0.
        for j in range(q.shape[1]):
            expected = -1.0 if (j + 1) % 2 == 1 else 1.0
            if q[0, j] * expected < 0:
                q[:, j] = -q[:, j]
        suffix_for_degree = {1: '.Linear', 2: '.Quadratic', 3: '.Cubic'}
        suffixes = [suffix_for_degree.get(d, f'^{d}') for d in range(1, n)]
        return ContrastMatrix(q, suffixes)


class Helmert:
    """Forward Helmert contrasts (level k+1 vs. mean of levels 1..k)."""

    def code_without_intercept(self, levels: Sequence) -> ContrastMatrix:
        n = len(levels)
        if n < 2:
            return _empty(n)
        m = np.full((n, n - 1), -1.0)
        for j in range(n - 1):
            m[j + 2 :, j] = 0.0
            m[j + 1, j] = j + 1
        suffixes = [f'[H.{i}]' for i in range(1, n)]
        return ContrastMatrix(m, suffixes)


class Diff:
    """Backward difference contrasts (level k vs. level k-1)."""

    def code_without_intercept(self, levels: Sequence) -> ContrastMatrix:
        n = len(levels)
        if n < 2:
            return _empty(n)
        m = np.empty((n, n - 1))
        for j in range(n - 1):
            m[: j + 1, j] = -(n - j - 1) / n
            m[j + 1 :, j] = (j + 1) / n
        suffixes = [f'[D.{i}]' for i in range(n - 1)]
        return ContrastMatrix(m, suffixes)


class Sum:
    """Sum (deviation) contrasts (level k vs. grand mean)."""

    def code_without_intercept(self, levels: Sequence) -> ContrastMatrix:
        n = len(levels)
        if n < 2:
            return _empty(n)
        m = np.zeros((n, n - 1))
        for i in range(n - 1):
            m[i, i] = 1.0
        m[n - 1, :] = -1.0
        suffixes = [f'[S.{i}]' for i in range(n - 1)]
        return ContrastMatrix(m, suffixes)
