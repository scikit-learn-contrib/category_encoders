"""Tests for the BinaryEncoder."""
from unittest import TestCase  # or `from unittest import ...` if on Python 3.4+

import category_encoders as encoders
import numpy as np
import pandas as pd


class TestBinaryEncoder(TestCase):
    """Unit tests for the BinaryEncoder."""

    def test_binary_bin(self):
        """Test the BinaryEncoder with only two values."""
        data = np.array(['a', 'ba', 'ba'])
        out = encoders.BinaryEncoder().fit_transform(data)
        self.assertTrue(pd.DataFrame([[0, 1], [1, 0], [1, 0]], columns=['0_0', '0_1']).equals(out))

    def test_binary_dist(self):
        """Test the BinaryEncoder with a all distinct values."""
        data = np.array(['apple', 'orange', 'peach', 'lemon'])
        encoder = encoders.BinaryEncoder()
        encoder.fit(data)

        # split dataframe into two transforms and recombine
        a = encoder.transform(data[:1])
        b = encoder.transform(data[1:])
        split = pd.concat([a, b])
        split = split.reset_index(drop=True)

        # run all at once
        c = encoder.transform(data)

        # make sure they are the same
        self.assertTrue(split.equals(c))
