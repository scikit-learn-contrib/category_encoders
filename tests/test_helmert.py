"""Tests for the HelmertEncoder."""
from unittest import TestCase  # or `from unittest import ...` if on Python 3.4+

import category_encoders as encoders
import numpy as np


class TestHelmertEncoder(TestCase):
    """Unit tests for the HelmertEncoder."""

    def test_get_contrast_matrix(self):
        """Should return the correct contrast matrix for helmert."""
        train = np.array([('A', ), ('B', ), ('C', )])
        encoder = encoders.HelmertEncoder()
        matrix = encoder.get_contrast_matrix(train)
        expected_matrix = np.array([[-1, -1], [1, -1], [0, 2]])
        np.testing.assert_array_equal(matrix.matrix, expected_matrix)
