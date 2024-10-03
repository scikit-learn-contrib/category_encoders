"""Tests for the BackwardDifferenceEncoder."""
from unittest import TestCase  # or `from unittest import ...` if on Python 3.4+

import category_encoders as encoders
import numpy as np


class TestBackwardsEncoder(TestCase):
    """Unit tests for the BackwardDifferenceEncoder."""

    def test_get_contrast_matrix(self):
        """Test the BackwardDifferenceEncoder get_contrast_matrix method."""
        train = np.array([('A', ), ('B', ), ('C', )])
        encoder = encoders.BackwardDifferenceEncoder()
        matrix = encoder.get_contrast_matrix(train)
        expected_matrix = np.array([[-2/3, -1/3], [1/3, -1/3], [1/3, 2/3]])
        np.testing.assert_array_equal(matrix.matrix, expected_matrix)
