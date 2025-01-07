"""Tests for the MEstimateEncoder."""
from unittest import TestCase  # or `from unittest import ...` if on Python 3.4+

import category_encoders as encoders

x = ['A', 'A', 'B', 'B']
y = [1, 1, 0, 1]
x_t = ['A', 'B', 'C']


class TestMEstimateEncoder(TestCase):
    """Tests for the MEstimateEncoder."""

    def test_reference_m0(self):
        """Test the MEstimateEncoder with m=0, i.e. no shrinking."""
        encoder = encoders.MEstimateEncoder(m=0, handle_unknown='value', handle_missing='value')
        encoder.fit(x, y)
        scored = encoder.transform(x_t)

        expected = [[1], [0.5], [3.0 / 4.0]]  # The prior probability
        self.assertEqual(scored.to_numpy().tolist(), expected)

    def test_reference_m1(self):
        """Test the MEstimateEncoder with m=1."""
        encoder = encoders.MEstimateEncoder(m=1, handle_unknown='value', handle_missing='value')
        encoder.fit(x, y)
        scored = encoder.transform(x_t)

        expected = [
            [(2 + 3.0 / 4.0) / (2 + 1)],
            [(1 + 3.0 / 4.0) / (2 + 1)],
            [3.0 / 4.0],
        ]
        self.assertEqual(scored.to_numpy().tolist(), expected)
