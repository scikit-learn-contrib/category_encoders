"""Tests for the Ordinal encoder."""
from unittest import TestCase  # or `from unittest import ...` if on Python 3.4+

import category_encoders as encoders
import numpy as np
import pandas as pd

import tests.helpers as th

np_X = th.create_array(n_rows=100)
np_X_t = th.create_array(n_rows=50, extras=True)
np_y = np.random.randn(np_X.shape[0]) > 0.5
np_y_t = np.random.randn(np_X_t.shape[0]) > 0.5
X = th.create_dataset(n_rows=100)
X_t = th.create_dataset(n_rows=50, extras=True)
y = pd.DataFrame(np_y)
y_t = pd.DataFrame(np_y_t)


class TestOrdinalEncoder(TestCase):
    """Unit tests for the Ordinal encoder."""

    def test_ordinal(self):
        """Test some basic functionality."""
        enc = encoders.OrdinalEncoder(verbose=1, return_df=True)
        enc.fit(X)
        out = enc.transform(X_t)
        self.assertEqual(len(set(out['extra'].values)), 4)
        self.assertIn(-1, set(out['extra'].values))
        self.assertFalse(enc.mapping is None)
        self.assertTrue(len(enc.mapping) > 0)

        enc = encoders.OrdinalEncoder(verbose=1, mapping=enc.mapping, return_df=True)
        enc.fit(X)
        out = enc.transform(X_t)
        self.assertEqual(len(set(out['extra'].values)), 4)
        self.assertIn(-1, set(out['extra'].values))
        self.assertTrue(len(enc.mapping) > 0)

        enc = encoders.OrdinalEncoder(verbose=1, return_df=True, handle_unknown='return_nan')
        enc.fit(X)
        out = enc.transform(X_t)
        out_cats = [x for x in set(out['extra'].values) if np.isfinite(x)]
        self.assertEqual(len(out_cats), 3)
        self.assertFalse(enc.mapping is None)

    def test_ordinal_dist(self):
        """Test that the encoder works with multiple columns and all encodings are distinct."""
        data = np.array([['apple', 'lemon'], ['peach', None]])
        encoder = encoders.OrdinalEncoder()
        result = encoder.fit_transform(data)
        self.assertEqual(2, len(result[0].unique()))
        self.assertEqual(2, len(result[1].unique()))
        self.assertFalse(np.isnan(result.iloc[1, 1]))

        encoder = encoders.OrdinalEncoder(handle_missing='return_nan')
        result = encoder.fit_transform(data)
        self.assertEqual(2, len(result[0].unique()))
        self.assertEqual(2, len(result[1].unique()))

    def test_pandas_categorical(self):
        """Test that the encoder works with pandas Categorical data."""
        X = pd.DataFrame(
            {
                'Str': ['a', 'c', 'c', 'd'],
                'Categorical': pd.Categorical(
                    list('bbea'), categories=['e', 'a', 'b'], ordered=True
                ),
            }
        )

        enc = encoders.OrdinalEncoder()
        out = enc.fit_transform(X)

        th.verify_numeric(out)
        self.assertEqual(3, out['Categorical'][0])
        self.assertEqual(3, out['Categorical'][1])
        self.assertEqual(1, out['Categorical'][2])
        self.assertEqual(2, out['Categorical'][3])

    def test_handle_missing_have_nan_fit_time_expect_as_category(self):
        """Test that missing values are encoded with 1 if handle_missing='value'."""
        train = pd.DataFrame(
            {
                'city': ['chicago', np.nan],
                'city_cat': pd.Categorical(['chicago', np.nan]),
            }
        )

        enc = encoders.OrdinalEncoder(handle_missing='value')
        out = enc.fit_transform(train)

        self.assertListEqual([1, 2], out['city'].tolist())
        self.assertListEqual([1, 2], out['city_cat'].tolist())

    def test_handle_missing_have_nan_transform_time_expect_negative_2(self):
        """Test that missing values in the test set are encoded with -2 if no missing in training.

        This is for handle_missing='value'.
        """
        train = pd.DataFrame(
            {
                'city': ['chicago', 'st louis'],
                'city_cat': pd.Categorical(['chicago', 'st louis']),
            }
        )
        test = pd.DataFrame(
            {
                'city': ['chicago', np.nan],
                'city_cat': pd.Categorical(['chicago', np.nan]),
            }
        )

        enc = encoders.OrdinalEncoder(handle_missing='value')
        enc.fit(train)
        out = enc.transform(test)

        self.assertListEqual([1, -2], out['city'].tolist())
        self.assertListEqual([1, -2], out['city_cat'].tolist())

    def test_handle_unknown_have_new_value_expect_negative_1(self):
        """Test that unknown values are encoded with -1 if missing values are left missing."""
        # See issue #238
        train = pd.DataFrame({'city': ['chicago', 'st louis']})
        test = pd.DataFrame({'city': ['chicago', 'los angeles']})
        expected = [1.0, -1.0]

        enc = encoders.OrdinalEncoder(handle_missing='return_nan')
        enc.fit(train)
        result = enc.transform(test)['city'].tolist()

        self.assertEqual(expected, result)

    def test_handle_unknown_have_new_value_expect_negative_1_categorical(self):
        """Test that unknown values are encoded with -1."""
        cities = ['st louis', 'chicago', 'los angeles']
        train = pd.DataFrame({'city': pd.Categorical(cities[:-1], categories=cities)})
        test = pd.DataFrame({'city': pd.Categorical(cities[1:], categories=cities)})

        expected = [2.0, -1.0]

        enc = encoders.OrdinalEncoder(handle_missing='return_nan')
        enc.fit(train)
        result = enc.transform(test)['city'].tolist()

        self.assertEqual(expected, result)

    def test_custom_mapping(self):
        """Test that custom mapping is correctly applied."""
        # See issue 193
        custom_mapping = [
            {
                'col': 'col1',
                'mapping': {np.nan: 0, 'a': 1, 'b': 2},
            },  # The mapping from the documentation
            {'col': 'col2', 'mapping': {np.nan: -3, 'x': 11, 'y': 2}},
        ]
        custom_mapping_series = [
            {
                'col': 'col1',
                'mapping': pd.Series({np.nan: 0, 'a': 1, 'b': 2}),
            },  # The mapping from the documentation
            {'col': 'col2', 'mapping': pd.Series({np.nan: -3, 'x': 11, 'y': 2})},
        ]

        train = pd.DataFrame({'col1': ['a', 'a', 'b', np.nan], 'col2': ['x', 'y', np.nan, np.nan]})

        for mapping in [custom_mapping, custom_mapping_series]:
            with self.subTest():
                enc = encoders.OrdinalEncoder(handle_missing='value', mapping=mapping)
                # We have to first 'fit' before 'transform'
                out = enc.fit_transform(
                    train
                )

                self.assertListEqual([1, 1, 2, 0], out['col1'].tolist())
                self.assertListEqual([11, 2, -3, -3], out['col2'].tolist())

    def test_integers_are_encoded(self):
        """Should encode integers, also negative ones as categories."""
        train = pd.DataFrame({'city': [-1]})
        expected = [1]

        enc = encoders.OrdinalEncoder(cols=['city'])
        result = enc.fit_transform(train)['city'].tolist()

        self.assertEqual(expected, result)

    def test_nan_in_training(self):
        """Test that NaN values are encoded the same way as non-missing the default setting."""
        train = pd.DataFrame({'city': [np.nan]})
        expected = [1]

        enc = encoders.OrdinalEncoder(cols=['city'])
        result = enc.fit_transform(train)['city'].tolist()

        self.assertEqual(expected, result)

    def test_timestamp(self):
        """Test that the ordinal encoder works with pandas timestamps."""
        df = pd.DataFrame(
            {
                'timestamps': {
                    0: pd.Timestamp('1997-09-03 00:00:00'),
                    1: pd.Timestamp('1997-09-03 00:00:00'),
                    2: pd.Timestamp('2000-09-03 00:00:00'),
                    3: pd.Timestamp('1997-09-03 00:00:00'),
                    4: pd.Timestamp('1999-09-04 00:00:00'),
                    5: pd.Timestamp('2001-09-03 00:00:00'),
                },
            }
        )
        enc = encoders.OrdinalEncoder(cols=['timestamps'])
        encoded_df = enc.fit_transform(df)
        expected_index = [
            pd.Timestamp('1997-09-03 00:00:00'),
            pd.Timestamp('2000-09-03 00:00:00'),
            pd.Timestamp('1999-09-04 00:00:00'),
            pd.Timestamp('2001-09-03 00:00:00'),
            pd.NaT,
        ]
        expected_mapping = pd.Series([1, 2, 3, 4, -2], index=expected_index)
        expected_values = [1, 1, 2, 1, 3, 4]

        pd.testing.assert_series_equal(expected_mapping, enc.mapping[0]['mapping'])
        self.assertListEqual(expected_values, encoded_df['timestamps'].tolist())

    def test_no_gaps(self):
        """Test that the ordinal mapping does not have gaps."""
        train = pd.DataFrame({'city': ['New York', np.nan, 'Rio', None, 'Rosenheim']})
        expected_mapping_value = pd.Series(
            [1, 2, 3, 4], index=['New York', 'Rio', 'Rosenheim', np.nan]
        )
        expected_mapping_return_nan = pd.Series(
            [1, 2, 3, -2], index=['New York', 'Rio', 'Rosenheim', np.nan]
        )

        enc_value = encoders.OrdinalEncoder(cols=['city'], handle_missing='value')
        enc_value.fit(train)
        pd.testing.assert_series_equal(expected_mapping_value, enc_value.mapping[0]['mapping'])
        enc_return_nan = encoders.OrdinalEncoder(cols=['city'], handle_missing='return_nan')
        enc_return_nan.fit(train)
        pd.testing.assert_series_equal(
            expected_mapping_return_nan, enc_return_nan.mapping[0]['mapping']
        )

    def test_nan_and_none_is_encoded_the_same(self):
        """Test that NaN and None are encoded the same."""
        train = pd.DataFrame({'city': [np.nan, None]})
        expected = [1, 1]

        enc = encoders.OrdinalEncoder(cols=['city'])
        result = enc.fit_transform(train)['city'].tolist()

        self.assertEqual(expected, result)

        new_nan = pd.DataFrame(
            {
                'city': [
                    np.nan,
                ]
            }
        )
        result_new_nan = enc.transform(new_nan)['city'].tolist()
        expected_new_nan = [1]
        self.assertEqual(expected_new_nan, result_new_nan)

        new_none = pd.DataFrame(
            {
                'city': [
                    None,
                ]
            }
        )
        result_new_none = enc.transform(new_none)['city'].tolist()
        expected_new_none = [1]
        self.assertEqual(expected_new_none, result_new_none)

    def test_inverse_transform_unknown_value(self):
        """Test the inverse transform with handle_unknown='value'.

        This should raise a warning as the unknown category cannot be inverted.
        """
        train = pd.DataFrame({'city': ['chicago', 'st louis']})
        test = pd.DataFrame({'city': ['chicago', 'los angeles']})

        enc = encoders.OrdinalEncoder(handle_missing='value', handle_unknown='value')
        enc.fit(train)
        result = enc.transform(test)

        message = (
            'inverse_transform is not supported because transform impute '
            'the unknown category -1 when encode city'
        )

        with self.assertWarns(UserWarning, msg=message):
            enc.inverse_transform(result)

    def test_inverse_transform_missing_value( self ):
        """Test the inverse transform with handle_missing='value'.

        This should output the original data if the input data is inverse transformed.
        """
        train = pd.DataFrame({'city': ['chicago', np.nan]})

        enc = encoders.OrdinalEncoder(handle_missing='value', handle_unknown='value')
        result = enc.fit_transform(train)
        original = enc.inverse_transform(result)

        pd.testing.assert_frame_equal(train, original)

    def test_inverse_transform_missing_return_nan(self):
        """Test the inverse transform with handle_missing='return_nan'.

        This should output the original data if the input data is inverse transformed.
        """
        train = pd.DataFrame({'city': ['chicago', np.nan]})

        enc = encoders.OrdinalEncoder(handle_missing='return_nan', handle_unknown='value')
        result = enc.fit_transform(train)
        original = enc.inverse_transform(result)

        pd.testing.assert_frame_equal(train, original)

    def test_inverse_transform_missing_and_unknown_return_nan(self):
        """Test the inverse transform with handle_missing and handle_unknown='return_nan'.

        This should raise a warning as the unknown category cannot be inverted.
        """
        train = pd.DataFrame({'city': ['chicago', np.nan]})
        test = pd.DataFrame({'city': ['chicago', 'los angeles']})

        enc = encoders.OrdinalEncoder(handle_missing='return_nan', handle_unknown='return_nan')
        enc.fit(train)
        result = enc.transform(test)

        message = (
            'inverse_transform is not supported because transform impute '
            'the unknown category nan when encode city'
        )

        with self.assertWarns(UserWarning, msg=message):
            enc.inverse_transform(result)

    def test_inverse_transform_handle_missing_value(self):
        """Test that the inverse transform works with handle_missing='value'."""
        train = pd.DataFrame({'city': ['chicago', np.nan]})
        enc = encoders.OrdinalEncoder(handle_missing='value', handle_unknown='return_nan')
        enc.fit(train)
        with self.subTest("Should treat unknown values as NaN values in the inverse."):
            test = pd.DataFrame({'city': ['chicago', 'los angeles']})
            result = enc.transform(test)
            original = enc.inverse_transform(result)
            pd.testing.assert_frame_equal(train, original)

        with self.subTest("Should treat unknown and NaN values as NaN in the inverse."):
            test = pd.DataFrame({'city': ['chicago', np.nan, 'los angeles']})
            expected = pd.DataFrame({'city': ['chicago', np.nan, np.nan]})
            result = enc.transform(test)
            original = enc.inverse_transform(result)
            pd.testing.assert_frame_equal(expected, original)

    def test_inverse_with_mapping(self):
        """Test that the inverse transform works with a custom mapping."""
        df = X.copy(deep=True)
        categoricals = [
            'unique_int',
            'unique_str',
            'invariant',
            'underscore',
            'none',
            'extra',
        ]
        mappings = {
            'as Series': [
                {
                    'col': c,
                    'mapping': pd.Series(data=range(len(df[c].unique())), index=df[c].unique()),
                    'data_type': X[c].dtype,
                }
                for c in categoricals
            ],
            'as Dict': [
                {'col': c, 'mapping': {k: idx for idx, k in enumerate(df[c].unique())}}
                for c in categoricals
            ],
        }
        for msg, mapping in mappings.items():
            with self.subTest(msg):
                df = X.copy(deep=True)
                enc = encoders.OrdinalEncoder(
                    cols=categoricals,
                    handle_unknown='ignore',
                    mapping=mapping,
                    return_df=True,
                )
                df[categoricals] = enc.fit_transform(df[categoricals])
                recovered = enc.inverse_transform(df[categoricals])
                pd.testing.assert_frame_equal(X[categoricals], recovered)

    def test_validate_mapping(self):
        """Test that the mapping is validated correctly."""
        custom_mapping = [
            {
                'col': 'col1',
                'mapping': {np.nan: 0, 'a': 1, 'b': 2},
            },  # The mapping from the documentation
            {'col': 'col2', 'mapping': {np.nan: -3, 'x': 11, 'y': 2}},
        ]
        expected_valid_mapping = [
            {
                'col': 'col1',
                'mapping': pd.Series({np.nan: 0, 'a': 1, 'b': 2}),
            },  # The mapping from the documentation
            {'col': 'col2', 'mapping': pd.Series({np.nan: -3, 'x': 11, 'y': 2})},
        ]
        enc = encoders.OrdinalEncoder()
        actual_valid_mapping = enc._validate_supplied_mapping(custom_mapping)
        self.assertEqual(len(actual_valid_mapping), len(expected_valid_mapping))
        for idx in range(len(actual_valid_mapping)):
            self.assertEqual(actual_valid_mapping[idx]['col'], expected_valid_mapping[idx]['col'])
            pd.testing.assert_series_equal(
                actual_valid_mapping[idx]['mapping'], expected_valid_mapping[idx]['mapping']
            )
