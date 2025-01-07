"""Tests for the utils module."""
from unittest import TestCase  # or `from unittest import ...` if on Python 3.4+

import numpy as np
import pandas as pd
import pytest
from category_encoders.utils import (
    BaseEncoder,
    convert_input_vector,
    convert_inputs,
    get_categorical_cols,
)
from packaging.version import Version
from sklearn import __version__ as skl_version
from sklearn.base import BaseEstimator, TransformerMixin


class TestUtils(TestCase):
    """Tests for the utils module."""

    def test_convert_input_vector(self):
        """Test the convert_input_vector function."""
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
        np.testing.assert_array_equal(
            result.index, [4, 5, 6], 'We want to preserve the original index'
        )

        result = convert_input_vector(
            pd.DataFrame({'y': [0, 1, 0]}, index=[4, 5, 6]), index
        )  # dataFrame
        self.assertTrue(isinstance(result, pd.Series))
        self.assertEqual(3, len(result))
        np.testing.assert_array_equal(
            result.index, [4, 5, 6], 'We want to preserve the original index'
        )

        result = convert_input_vector((0, 1, 0), index)  # tuple
        self.assertTrue(isinstance(result, pd.Series))
        self.assertEqual(3, len(result))
        np.testing.assert_array_equal(result.index, [2, 3, 4])

        # should not work for scalars
        self.assertRaises(ValueError, convert_input_vector, 0, [2])
        self.assertRaises(ValueError, convert_input_vector, "a", [2])

        # multiple columns and rows should cause an error because it is unclear
        # which column/row to use as the target
        self.assertRaises(
            ValueError,
            convert_input_vector,
            (pd.DataFrame({'col1': [0, 1, 0], 'col2': [1, 0, 1]})),
            index,
        )
        self.assertRaises(
            ValueError, convert_input_vector, (np.array([[0, 1], [1, 0], [0, 1]])), index
        )
        self.assertRaises(ValueError, convert_input_vector, ([[0, 1], [1, 0], [0, 1]]), index)

        # edge scenarios (it is ok to raise an exception but please,
        # provide then a helpful exception text)
        _ = convert_input_vector(pd.Series(dtype=float), [])
        _ = convert_input_vector([], [])
        _ = convert_input_vector([[]], [])
        _ = convert_input_vector(pd.DataFrame(), [])

    def test_convert_inputs(self):
        """Test the convert_inputs function."""
        aindex = [2, 4, 5]
        bindex = [1, 3, 4]
        alist = [5, 3, 6]
        aseries = pd.Series(alist, aindex)
        barray = np.array([[7, 9], [4, 3], [0, 1]])
        bframe = pd.DataFrame(barray, bindex)

        X, y = convert_inputs(barray, alist)
        self.assertTrue(isinstance(X, pd.DataFrame))
        self.assertTrue(isinstance(y, pd.Series))
        self.assertEqual((3, 2), X.shape)
        self.assertEqual(3, len(y))
        self.assertTrue(list(X.index) == list(y.index) == [0, 1, 2])

        X, y = convert_inputs(barray, alist, index=aindex)
        self.assertTrue(isinstance(X, pd.DataFrame))
        self.assertTrue(isinstance(y, pd.Series))
        self.assertEqual((3, 2), X.shape)
        self.assertEqual(3, len(y))
        self.assertTrue(list(X.index) == list(y.index) == aindex)

        X, y = convert_inputs(barray, aseries, index=bindex)
        self.assertTrue(isinstance(X, pd.DataFrame))
        self.assertTrue(isinstance(y, pd.Series))
        self.assertEqual((3, 2), X.shape)
        self.assertEqual(3, len(y))
        self.assertTrue(list(X.index) == list(y.index) == aindex)

        X, y = convert_inputs(bframe, alist, index=[3, 1, 4])
        self.assertTrue(isinstance(X, pd.DataFrame))
        self.assertTrue(isinstance(y, pd.Series))
        self.assertEqual((3, 2), X.shape)
        self.assertEqual(3, len(y))
        self.assertTrue(list(X.index) == list(y.index) == bindex)

        self.assertRaises(ValueError, convert_inputs, bframe, aseries)

        # shape mismatch
        self.assertRaises(ValueError, convert_inputs, barray, [1, 2, 3, 4])

    def test_get_categorical_cols(self):
        """Test the get_categorical_cols function."""
        df = pd.DataFrame({'col': ['a', 'b']})
        self.assertEqual(get_categorical_cols(df.astype('object')), ['col'])
        self.assertEqual(get_categorical_cols(df.astype('category')), ['col'])
        self.assertEqual(get_categorical_cols(df.astype('string')), ['col'])


class TestBaseEncoder(TestCase):
    """Tests for the BaseEncoder class."""

    def setUp(self):
        """Set up the tests."""
        class DummyEncoder(BaseEncoder, BaseEstimator, TransformerMixin):
            def _fit(self, X, y=None):
                return self

            def transform(self, X, y=None, override_return_df=False):
                return X

        self.encoder = DummyEncoder()

    @pytest.mark.skipif(Version(skl_version) < Version('1.2'), reason='requires sklean > 1.2')
    def test_sklearn_pandas_out_refit(self):
        """Test that the encoder can be refit with sklearn and pandas."""
        # Thanks to Issue#437
        df = pd.DataFrame({'C1': ['a', 'a'], 'C2': ['c', 'd']})
        self.encoder.set_output(transform='pandas')
        self.encoder.fit_transform(df.iloc[:1])
        out = self.encoder.fit_transform(df.rename(columns={'C1': 'X1', 'C2': 'X2'}))
        self.assertTrue(list(out.columns) == ['X1', 'X2'])
