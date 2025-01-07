"""Tests for the OneHotEncoder."""
from unittest import TestCase  # or `from unittest import ...` if on Python 3.4+

import category_encoders as encoders
import numpy as np
import pandas as pd

import tests.helpers as th


class TestOneHotEncoder(TestCase):
    """Tests for the OneHotEncoder."""

    def test_one_hot(self):
        """Test basic functionality."""
        X = th.create_dataset(n_rows=100)
        X_t = th.create_dataset(n_rows=50, extras=True)
        enc = encoders.OneHotEncoder(verbose=1, return_df=False)
        enc.fit(X)
        self.assertEqual(
            enc.transform(X_t).shape[1],
            enc.transform(X).shape[1],
            'We have to get the same count of columns despite the presence of a new value',
        )

        enc = encoders.OneHotEncoder(verbose=1, return_df=True, handle_unknown='indicator')
        enc.fit(X)
        out = enc.transform(X_t)
        self.assertIn('extra_-1', out.columns)

        enc = encoders.OneHotEncoder(verbose=1, return_df=True, handle_unknown='return_nan')
        enc.fit(X)
        out = enc.transform(X_t)
        self.assertEqual(len([x for x in out.columns if str(x).startswith('extra_')]), 3)

        enc = encoders.OneHotEncoder(verbose=1, return_df=True, handle_unknown='error')
        # The exception is already raised in fit() because transform() is called there to get
        # feature_names right.
        enc.fit(X)
        with self.assertRaises(ValueError):
            enc.transform(X_t)

        enc = encoders.OneHotEncoder(
            verbose=1, return_df=True, handle_unknown='return_nan', use_cat_names=True
        )
        enc.fit(X)
        out = enc.transform(X_t)
        self.assertIn('extra_A', out.columns)

        enc = encoders.OneHotEncoder(
            verbose=1, return_df=True, use_cat_names=True, handle_unknown='indicator'
        )
        enc.fit(X)
        out = enc.transform(X_t)
        self.assertIn('extra_-1', out.columns)

        # test inverse_transform
        X_i = th.create_dataset(n_rows=100, has_missing=False)
        X_i_t = th.create_dataset(n_rows=50, has_missing=False)
        cols = ['underscore', 'none', 'extra', 'categorical']

        enc = encoders.OneHotEncoder(verbose=1, use_cat_names=True, cols=cols)
        enc.fit(X_i)
        obtained = enc.inverse_transform(enc.transform(X_i_t))
        th.verify_inverse_transform(X_i_t, obtained)

    def test_fit_transform_use_cat_names(self):
        """Test that use_cat_names works as expected.

        @ToDo: This test is not very useful as it seems to be covered by other tests already.
        """
        encoder = encoders.OneHotEncoder(
            cols=[0], use_cat_names=True, handle_unknown='indicator', return_df=False
        )

        result = encoder.fit_transform([[-1]])

        self.assertListEqual([[1, 0]], result.tolist())

    def test_inverse_transform_duplicated_cat_names(self):
        """Test that inverse_transform works with duplicated cat names.

        This can happen if use_cat_names is true and the two new column names coincide because
        col_1 + label_A is the lame as col_2 + label_B.
        """
        cases = {"should work if use_cat_names is True": True,
                 "should work if use_cat_names is False": False}
        for case, use_cat_names in cases.items():
            with self.subTest(case=case):
                encoder = encoders.OneHotEncoder(cols=['match', 'match_box'],
                                                 use_cat_names=use_cat_names)
                value = pd.DataFrame({'match': pd.Series('box_-1'), 'match_box': pd.Series(-1)})

                transformed = encoder.fit_transform(value)
                inverse_transformed = encoder.inverse_transform(transformed)

                pd.testing.assert_frame_equal(value, inverse_transformed)

    def test_fit_transform_duplicated_column_rename(self):
        """Check that # is added to duplicated column names.

        Column names can be duplicated either by use_cat_names=True or by having the label -1
        and adding an indicator column.
        """
        encoder = encoders.OneHotEncoder(
            cols=['match', 'match_box'], use_cat_names=True, handle_unknown='indicator'
        )
        value = pd.DataFrame({'match': pd.Series('box_-1'), 'match_box': pd.Series('-1')})

        result = encoder.fit_transform(value)
        columns = result.columns.tolist()

        self.assertSetEqual(
            {'match_box_-1', 'match_-1', 'match_box_-1#', 'match_box_-1##'}, set(columns)
        )

    def test_fit_transform_handle_unknown_value(self):
        """Test that unseen values are encoded as all zeroes."""
        train = pd.DataFrame({'city': ['Chicago', 'Seattle']})
        enc = encoders.OneHotEncoder(handle_unknown='value')
        enc.fit(train)
        with self.subTest("should encode unseen values as all zeroes"):
            test = pd.DataFrame({'city': ['Chicago', 'Detroit']})
            expected_result = pd.DataFrame(
                {'city_1': [1, 0], 'city_2': [0, 0]}, columns=['city_1', 'city_2']
            )
            result = enc.transform(test)
            pd.testing.assert_frame_equal(expected_result, result)

        with self.subTest("should work if no unseen data"):
            expected_result = pd.DataFrame(
                {'city_1': [1, 0], 'city_2': [0, 1]}, columns=['city_1', 'city_2']
            )
            result = enc.transform(train)
            pd.testing.assert_frame_equal(expected_result, result)

    def test_fit_transform_handle_unknown_indicator(self):
        """Test that unseen values are encoded with an indicator column."""
        train = pd.DataFrame({'city': ['Chicago', 'Seattle']})
        enc = encoders.OneHotEncoder(handle_unknown='indicator')
        enc.fit(train)
        with self.subTest("Should create a column even if no unseen value in transform stage"):
            expected_result = pd.DataFrame(
                {'city_1': [1, 0], 'city_2': [0, 1], 'city_-1': [0, 0]},
                columns=['city_1', 'city_2', 'city_-1'],
            )
            result = enc.transform(train)
            pd.testing.assert_frame_equal(expected_result, result)
        with self.subTest("Should create a column if unseen value in transform stage"):

            test = pd.DataFrame({'city': ['Chicago', 'Detroit']})
            expected_result = pd.DataFrame(
                {'city_1': [1, 0], 'city_2': [0, 0], 'city_-1': [0, 1]},
                columns=['city_1', 'city_2', 'city_-1'],
            )
            result = enc.transform(test)
            pd.testing.assert_frame_equal(expected_result, result)

    def test_handle_missing_error(self):
        """Test that missing values raise an error."""
        data_no_missing = ['A', 'B', 'B']
        data_w_missing = [np.nan, 'B', 'B']
        encoder = encoders.OneHotEncoder(handle_missing='error')

        result = encoder.fit_transform(data_no_missing)
        expected = [[1, 0], [0, 1], [0, 1]]
        self.assertEqual(result.to_numpy().tolist(), expected)

        self.assertRaisesRegex(ValueError, '.*null.*', encoder.transform, data_w_missing)

        self.assertRaisesRegex(ValueError, '.*null.*', encoder.fit, data_w_missing)

    def test_handle_missing_return_nan(self):
        """Test that missing values are encoded as NaN in each dummy column."""
        train = pd.DataFrame({'x': ['A', np.nan, 'B']})
        encoder = encoders.OneHotEncoder(handle_missing='return_nan', use_cat_names=True)
        result = encoder.fit_transform(train)
        pd.testing.assert_frame_equal(
            result,
            pd.DataFrame({'x_A': [1, np.nan, 0], 'x_B': [0, np.nan, 1]}),
        )

    def test_handle_missing_ignore(self):
        """Test that missing values are encoded as 0 in each dummy column."""
        train = pd.DataFrame(
            {'x': ['A', 'B', np.nan], 'y': ['A', None, 'A'], 'z': [np.nan, 'B', 'B']}
        )
        train['z'] = train['z'].astype('category')

        expected_result = pd.DataFrame(
            {'x_A': [1, 0, 0], 'x_B': [0, 1, 0], 'y_A': [1, 0, 1], 'z_B': [0, 1, 1]}
        )
        encoder = encoders.OneHotEncoder(handle_missing='ignore', use_cat_names=True)
        result = encoder.fit_transform(train)

        pd.testing.assert_frame_equal(result, expected_result)

    def test_handle_missing_ignore_test_mapping(self):
        """Test that the mapping is correct if handle_missing='ignore'."""
        train = pd.DataFrame({'city': ['Chicago', np.nan, 'Geneva']})
        expected_result = pd.DataFrame({'city_1': [1, 0, 0], 'city_2': [0, 0, 1]})

        encoder = encoders.OneHotEncoder(handle_missing='ignore')
        result = encoder.fit(train).transform(train)
        expected_mapping = pd.DataFrame(
            [
                [1, 0],
                [0, 1],
                [0, 0],
                [0, 0],
            ],
            columns=['city_1', 'city_2'],
            index=[1, 2, -2, -1],
        )

        pd.testing.assert_frame_equal(expected_result, result)
        pd.testing.assert_frame_equal(expected_mapping, encoder.category_mapping[0]['mapping'])

    def test_handle_missing_indicator(self):
        """Test that missing values are encoded with an indicator column."""
        with self.subTest("Should create a column if NaN in training set"):
            train = ['A', 'B', np.nan]
            encoder = encoders.OneHotEncoder(handle_missing='indicator', handle_unknown='value')
            result = encoder.fit_transform(train)
            expected = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            self.assertEqual(result.to_numpy().tolist(), expected)

        with self.subTest("should create a column if NaN not in training set"):
            train = ['A', 'B']

            encoder = encoders.OneHotEncoder(handle_missing='indicator', handle_unknown='value')
            result = encoder.fit_transform(train)

            expected = [[1, 0, 0], [0, 1, 0]]
            self.assertEqual(result.to_numpy().tolist(), expected)

            # if NaN occurs in prediction it should be encoded as a new column
            test = ['A', 'B', np.nan]
            encoded_test = encoder.transform(test)
            expected_test = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            self.assertEqual(encoded_test.to_numpy().tolist(), expected_test)

    def test_handle_unknown_indicator(self):
        """Test that unseen values are encoded with an indicator column."""
        train = ['A', 'B']
        encoder = encoders.OneHotEncoder(handle_unknown='indicator', handle_missing='value')
        encoder.fit(train)
        with self.subTest("should create a column if unseen value in transform stage"):
            test = ['A', 'B', 'C']
            result = encoder.transform(test)
            expected = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            self.assertEqual(result.to_numpy().tolist(), expected)

        with self.subTest("should also create a column if no unseen value in transform"):
            result = encoder.transform(train)
            expected = [[1, 0, 0], [0, 1, 0]]
            self.assertEqual(result.to_numpy().tolist(), expected)

    def test_inverse_transform_missing_value(self):
        """Test the inverse transform with handle_missing='value'.

        This should output the original data if the input data is inverse transformed.
        """
        train = pd.DataFrame({'city': ['chicago', np.nan]})

        enc = encoders.OneHotEncoder(handle_missing='value', handle_unknown='value')
        result = enc.fit_transform(train)
        original = enc.inverse_transform(result)

        pd.testing.assert_frame_equal(train, original)

    def test_inverse_transform_missing_return_nan(self):
        """Test the inverse transform with handle_missing='return_nan'.

        This should output the original data if the input data is inverse transformed.
        """
        train = pd.DataFrame({'city': ['chicago', np.nan]})

        enc = encoders.OneHotEncoder(handle_missing='return_nan', handle_unknown='value')
        result = enc.fit_transform(train)
        original = enc.inverse_transform(result)

        pd.testing.assert_frame_equal(train, original)

    def test_inverse_transform_missing_and_unknown_return_nan(self):
        """Test the inverse transform with handle_missing and handle_unknown='return_nan'.

        This should raise a warning as the unknown category cannot be inverted.
        """
        train = pd.DataFrame({'city': ['chicago', np.nan]})
        test = pd.DataFrame({'city': ['chicago', 'los angeles']})

        enc = encoders.OneHotEncoder(handle_missing='return_nan', handle_unknown='return_nan')
        enc.fit(train)
        result = enc.transform(test)

        message = (
            'inverse_transform is not supported because transform impute '
            'the unknown category nan when encode city'
        )

        with self.assertWarns(UserWarning, msg=message):
            enc.inverse_transform(result)

    def test_inverse_transform_handle_missing_value(self):
        """Test inverse transform if missing values are encoded with strategy 'value'."""
        train = pd.DataFrame({'city': ['chicago', np.nan]})

        enc = encoders.OneHotEncoder(handle_missing='value', handle_unknown='return_nan')
        enc.fit(train)

        test_data_case_1 = pd.DataFrame({'city': ['chicago', 'los angeles']})
        test_data_case_2 = pd.DataFrame({'city': ['chicago', np.nan, 'los angeles']})
        expected_case_2 = pd.DataFrame({'city': ['chicago', np.nan, np.nan]})
        cases = {"should encode unknown into nan": (test_data_case_1, train),
                 "should encode unknown into nan and missing into nan": (test_data_case_2,
                                                                         expected_case_2),
                 }
        for case, (test_data, expected) in cases.items():
            with self.subTest(case=case):
                result = enc.transform(test_data)
                original = enc.inverse_transform(result)
                pd.testing.assert_frame_equal(expected, original)
