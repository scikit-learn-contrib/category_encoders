"""Tests for the helpers module."""
from unittest import TestCase  # or `from unittest import ...` if on Python 3.4+

import numpy as np
import pandas as pd

from tests.helpers import verify_numeric


class TestHelpers(TestCase):
    """Tests for the helpers module."""

    def test_is_numeric_pandas(self):
        """Whole numbers, regardless of the byte length, should not raise AssertionError."""
        X = pd.DataFrame(np.ones([5, 5]), dtype='int32')
        verify_numeric(pd.DataFrame(X))

        X = pd.DataFrame(np.ones([5, 5]), dtype='int64')
        verify_numeric(pd.DataFrame(X))

        # Strings should raise AssertionError
        X = pd.DataFrame([['a', 'b', 'c'], ['d', 'e', 'f']])
        with self.assertRaises(AssertionError):
            verify_numeric(pd.DataFrame(X))

    def test_is_numeric_numpy(self):
        """Whole numbers, regardless of the byte length, should not raise AssertionError."""
        X = np.ones([5, 5], dtype='int32')
        verify_numeric(pd.DataFrame(X))

        X = np.ones([5, 5], dtype='int64')
        verify_numeric(pd.DataFrame(X))

        # Floats
        X = np.ones([5, 5], dtype='float32')
        verify_numeric(pd.DataFrame(X))

        X = np.ones([5, 5], dtype='float64')
        verify_numeric(pd.DataFrame(X))

    def test_verify_raises_assertion_error_on_categories(self):
        """Categories should raise AssertionError."""
        X = pd.DataFrame([['a', 'b', 'c'], ['d', 'e', 'f']], dtype='category')
        with self.assertRaises(AssertionError):
            verify_numeric(pd.DataFrame(X))
