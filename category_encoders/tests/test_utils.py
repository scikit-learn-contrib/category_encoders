from unittest2 import TestCase  # or `from unittest import ...` if on Python 3.4+
from category_encoders.utils import convert_input_vector
import pandas as pd
import numpy as np


class TestUtils(TestCase):
    def test_convert_input_vector(self):
        index = [2, 3, 4]

        result = convert_input_vector([0, 1, 0], index)  # list
        self.assertTrue(isinstance(result, pd.Series))
        self.assertEqual(3, len(result))
        np.testing.assert_array_equal(result.index, [2, 3, 4])

        result = convert_input_vector([[0, 1, 0]], index)  # list of lists (row)
        self.assertTrue(isinstance(result, pd.Series))
        self.assertEqual(3, len(result))
        np.testing.assert_array_equal(result.index, [2, 3, 4])

        result = convert_input_vector([[0], [1], [0]], index)  # list of lists (column)
        self.assertTrue(isinstance(result, pd.Series))
        self.assertEqual(3, len(result))
        np.testing.assert_array_equal(result.index, [2, 3, 4])

        result = convert_input_vector(np.array([1, 0, 1]), index)  # np vector
        self.assertTrue(isinstance(result, pd.Series))
        self.assertEqual(3, len(result))
        np.testing.assert_array_equal(result.index, [2, 3, 4])

        result = convert_input_vector(np.array([[1, 0, 1]]), index)  # np matrix row
        self.assertTrue(isinstance(result, pd.Series))
        self.assertEqual(3, len(result))
        np.testing.assert_array_equal(result.index, [2, 3, 4])

        result = convert_input_vector(np.array([[1], [0], [1]]), index)  # np matrix column
        self.assertTrue(isinstance(result, pd.Series))
        self.assertEqual(3, len(result))
        np.testing.assert_array_equal(result.index, [2, 3, 4])

        result = convert_input_vector(pd.Series([0, 1, 0], index=[4, 5, 6]), index)  # series
        self.assertTrue(isinstance(result, pd.Series))
        self.assertEqual(3, len(result))
        np.testing.assert_array_equal(result.index, [4, 5, 6], 'We want to preserve the original index')

        result = convert_input_vector(pd.DataFrame({'y': [0, 1, 0]}, index=[4, 5, 6]), index)  # dataFrame
        self.assertTrue(isinstance(result, pd.Series))
        self.assertEqual(3, len(result))
        np.testing.assert_array_equal(result.index, [4, 5, 6], 'We want to preserve the original index')

        result = convert_input_vector((0, 1, 0), index)  # tuple
        self.assertTrue(isinstance(result, pd.Series))
        self.assertEqual(3, len(result))
        np.testing.assert_array_equal(result.index, [2, 3, 4])

        result = convert_input_vector(0, [2])  # scalar
        self.assertTrue(isinstance(result, pd.Series))
        self.assertEqual(1, len(result))
        self.assertTrue(result.index == [2])

        result = convert_input_vector('a', [2])  # scalar
        self.assertTrue(isinstance(result, pd.Series))
        self.assertEqual(1, len(result))
        self.assertTrue(result.index == [2])

        # multiple columns and rows should cause an error because it is unclear which column/row to use as the target
        self.assertRaises(ValueError, convert_input_vector, (pd.DataFrame({'col1': [0, 1, 0], 'col2': [1, 0, 1]})), index)
        self.assertRaises(ValueError, convert_input_vector, (np.array([[0, 1], [1, 0], [0, 1]])), index)
        self.assertRaises(ValueError, convert_input_vector, ([[0, 1], [1, 0], [0, 1]]), index)

        # edge scenarios (it is ok to raise an exception but please, provide then a helpful exception text)
        _ = convert_input_vector(pd.Series(dtype=float), [])
        _ = convert_input_vector([], [])
        _ = convert_input_vector([[]], [])
        _ = convert_input_vector(pd.DataFrame(), [])
