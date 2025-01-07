"""Tests for the encoders."""
import warnings
from copy import deepcopy
from datetime import timedelta
from unittest import TestCase

import category_encoders as encoders
import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal
from sklearn.compose import ColumnTransformer
from sklearn.utils.estimator_checks import (
    check_n_features_in,
    check_transformer_general,
    check_transformers_unfitted,
)

import tests.helpers as th

__author__ = 'willmcginnis'


# data definitions
np_X = th.create_array(n_rows=100)
np_X_t = th.create_array(n_rows=50, extras=True)
np_y = np.random.randn(np_X.shape[0]) > 0.5
np_y_t = np.random.randn(np_X_t.shape[0]) > 0.5
X = th.create_dataset(n_rows=100)
X_t = th.create_dataset(n_rows=50, extras=True)
y = pd.DataFrame(np_y)
y_t = pd.DataFrame(np_y_t)


# turn warnings like division by zero into errors
np.seterr(all='raise')

# turn non-Numpy warnings into errors
warnings.filterwarnings('error')


class TestEncoders(TestCase):
    """Tests for the encoders.

    This is more of functional and property-based testing than unit testing.
    """

    def test_np(self):
        """Test all encoders with numpy arrays as input."""
        for encoder_name in encoders.__all__:
            with self.subTest(encoder_name=encoder_name):
                # Encode a numpy array
                enc = getattr(encoders, encoder_name)()
                enc.fit(np_X, np_y)
                th.verify_numeric(enc.transform(np_X_t))

    def test_classification(self):
        """Perform some basic testing of all encoders.

        This includes running the pipeline on various data types and with different parameters.
        """
        for encoder_name in encoders.__all__:
            with self.subTest(encoder_name=encoder_name):
                cols = [
                    'unique_str',
                    'underscore',
                    'extra',
                    'none',
                    'invariant',
                    'categorical',
                    'na_categorical',
                    'categorical_int',
                ]

                enc = getattr(encoders, encoder_name)(cols=cols)
                enc.fit(X, np_y)
                th.verify_numeric(enc.transform(X_t))

                enc = getattr(encoders, encoder_name)(verbose=1)
                enc.fit(X, np_y)
                th.verify_numeric(enc.transform(X_t))

                enc = getattr(encoders, encoder_name)(drop_invariant=True)
                enc.fit(X, np_y)
                th.verify_numeric(enc.transform(X_t))

                enc = getattr(encoders, encoder_name)(return_df=False)
                enc.fit(X, np_y)
                self.assertTrue(isinstance(enc.transform(X_t), np.ndarray))
                self.assertEqual(
                    enc.transform(X_t).shape[0], X_t.shape[0], 'Row count must not change'
                )

                # encoders should be re-fittable (c.f. issue 122)
                X_a = pd.DataFrame(data=['1', '2', '2', '2', '2', '2'], columns=['col_a'])
                X_b = pd.DataFrame(
                    data=['1', '1', '1', '2', '2', '2'], columns=['col_b']
                )  # different values and name
                y_dummy = [True, False, True, False, True, False]
                enc = getattr(encoders, encoder_name)()
                enc.fit(X_a, y_dummy)
                enc.fit(X_b, y_dummy)
                th.verify_numeric(enc.transform(X_b))

    def test_deepcopy(self):
        """Generate instance of every encoder and test if it is deepcopy-able.

        See: https://github.com/scikit-learn-contrib/categorical-encoding/pull/194
        """
        for encoder_name in encoders.__all__:
            with self.subTest(encoder_name=encoder_name):
                enc = getattr(encoders, encoder_name)()
                _ = deepcopy(enc)

    def test_impact_encoders(self):
        """Test that supervised encoders use a target variable."""
        for encoder_name in encoders.__all__:
            enc = getattr(encoders, encoder_name)()
            if not enc._get_tags().get('supervised_encoder'):
                continue
            with self.subTest(encoder_name=encoder_name):
                # encode a numpy array and transform with the help of the target
                enc.fit(np_X, np_y)
                th.verify_numeric(enc.transform(np_X_t, np_y_t))

                # target is a DataFrame
                enc = getattr(encoders, encoder_name)()
                enc.fit(X, y)
                th.verify_numeric(enc.transform(X_t, y_t))

                # when we run transform(X, y) and there is a new value in X,
                # something is wrong and we raise an error
                enc = getattr(encoders, encoder_name)(handle_unknown='error', cols=['extra'])
                enc.fit(X, y)
                self.assertRaises(ValueError, enc.transform, (X_t, y_t))

    def test_error_handling(self):
        """Test that the encoder raises an error if the input is wrong."""
        for encoder_name in encoders.__all__:
            with self.subTest(encoder_name=encoder_name):
                # we exclude some columns
                X = th.create_dataset(n_rows=100)
                X = X.drop(['unique_str', 'none'], axis=1)
                X_t = th.create_dataset(n_rows=50, extras=True)
                X_t = X_t.drop(['unique_str', 'none'], axis=1)

                # illegal state, we have to first train the encoder...
                enc = getattr(encoders, encoder_name)()
                with self.assertRaises(ValueError):
                    enc.transform(X)

                # wrong count of attributes
                enc = getattr(encoders, encoder_name)()
                enc.fit(X, y)
                with self.assertRaises(ValueError):
                    enc.transform(X_t.iloc[:, 0:3])

                # no cols
                enc = getattr(encoders, encoder_name)(cols=[])
                enc.fit(X, y)
                self.assertTrue(enc.transform(X_t).equals(X_t))

    def test_handle_unknown_error(self):
        """The encoder should raise an error if there is a new value and handle_unknown='error'."""
        # BaseN has problems with None -> ignore None
        X = th.create_dataset(n_rows=100, has_missing=False)
        X_t = th.create_dataset(n_rows=50, extras=True, has_missing=False)

        # HashingEncoder supports new values by design -> excluded
        for encoder_name in set(encoders.__all__) - { 'HashingEncoder' }:
            with self.subTest(encoder_name=encoder_name):
                # new value during scoring
                enc = getattr(encoders, encoder_name)(handle_unknown='error')
                enc.fit(X, y)
                with self.assertRaises(ValueError):
                    _ = enc.transform(X_t)

    def test_handle_missing_error(self):
        """The encoder should raise an error if there is a NaN value and handle_missing='error'."""
        non_null = pd.DataFrame(
            {'city': ['chicago', 'los angeles'], 'color': ['red', np.nan]}
        )  # only 'city' column is going to be transformed
        has_null = pd.DataFrame({'city': ['chicago', np.nan], 'color': ['red', np.nan]})
        has_null_pd = pd.DataFrame(
            {'city': ['chicago', pd.NA], 'color': ['red', pd.NA]}, dtype='string'
        )
        y = pd.Series([1, 0])

        # HashingEncoder supports new values by design -> excluded
        for encoder_name in set(encoders.__all__) - { 'HashingEncoder' }:
            with self.subTest(encoder_name=encoder_name):
                enc = getattr(encoders, encoder_name)(handle_missing='error', cols='city')
                with self.assertRaises(ValueError):
                    enc.fit(has_null, y)

                with self.assertRaises(ValueError):
                    enc.fit(has_null_pd, y)

                enc.fit(
                    non_null, y
                )  # we raise an error only if a missing value is in one of the transformed columns

                with self.assertRaises(ValueError):
                    enc.transform(has_null)

    def test_handle_missing_error_2cols(self):
        """The encoder should raise an error if there is a NaN value and handle_missing='error'.

        See issue #213.
        This test covers the case of multiple columns.
        """
        non_null = pd.DataFrame(
            {'country': ['us', 'uk'], 'city': ['chicago', 'los angeles'], 'color': ['red', np.nan]}
        )  # only 'city' column is going to be transformed
        has_null = pd.DataFrame(
            {'country': ['us', 'uk'], 'city': ['chicago', np.nan], 'color': ['red', np.nan]}
        )
        y = pd.Series([1, 0])

        # HashingEncoder supports new values by design -> excluded
        for encoder_name in set(encoders.__all__) - { 'HashingEncoder' }:
            with self.subTest(encoder_name=encoder_name):
                enc = getattr(encoders, encoder_name)(
                    handle_missing='error', cols=['country', 'city']
                )
                with self.assertRaises(ValueError):
                    enc.fit(has_null, y)

                enc.fit(
                    non_null, y
                )  # we raise an error only if a missing value is in one of the transformed columns

                with self.assertRaises(ValueError):
                    enc.transform(has_null)

    def test_handle_unknown_return_nan(self):
        """Test that the encoder implements a handle_unknown='return_nan' strategy."""
        train = pd.DataFrame({'city': ['chicago', 'los angeles']})
        test = pd.DataFrame({'city': ['chicago', 'denver']})
        y = pd.Series([1, 0])

        for encoder_name in set(encoders.__all__) - {
            'HashingEncoder'
        }:  # HashingEncoder supports new values by design -> excluded
            with self.subTest(encoder_name=encoder_name):
                enc = getattr(encoders, encoder_name)(handle_unknown='return_nan')
                enc.fit(train, y)
                result = enc.transform(test).iloc[1, :]

                if len(result) == 1:
                    self.assertTrue(result.isna().all())
                else:
                    self.assertTrue(result[1:].isna().all())

    def test_handle_missing_return_nan_train(self):
        """Test that the encoder implements a handle_missing='return_nan' strategy."""
        X_np = pd.DataFrame({'city': ['chicago', 'los angeles', np.nan]})
        X_pd = pd.DataFrame({'city': ['chicago', 'los angeles', pd.NA]}, dtype='string')
        y = pd.Series([1, 0, 1])

        for encoder_name in set(encoders.__all__) - {
            'HashingEncoder'
        }:  # HashingEncoder supports new values by design -> excluded
            for X in (X_np, X_pd):
                with self.subTest(encoder_name=encoder_name):
                    enc = getattr(encoders, encoder_name)(handle_missing='return_nan')
                    result = enc.fit_transform(X, y).iloc[2, :]

                if len(result) == 1:
                    self.assertTrue(result.isna().all())
                else:
                    self.assertTrue(result[1:].isna().all())

    def test_handle_missing_return_nan_test(self):
        """Test that the encoder implements a handle_missing='return_nan' strategy."""
        X = pd.DataFrame({'city': ['chicago', 'los angeles', 'chicago']})
        X_np = pd.DataFrame({'city': ['chicago', 'los angeles', np.nan]})
        X_pd = pd.DataFrame({'city': ['chicago', 'los angeles', pd.NA]}, dtype='string')
        y = pd.Series([1, 0, 1])

        # HashingEncoder supports new values by design -> excluded
        for encoder_name in set(encoders.__all__) - { 'HashingEncoder' }:
            for X_na in (X_np, X_pd):
                with self.subTest(encoder_name=encoder_name):
                    enc = getattr(encoders, encoder_name)(handle_missing='return_nan')
                    result = enc.fit(X, y).transform(X_na).iloc[2, :]

                if len(result) == 1:
                    self.assertTrue(result.isna().all())
                else:
                    self.assertTrue(result[1:].isna().all())

    def test_handle_unknown_value(self):
        """Test that each encoder implements a handle_unknown='value' strategy."""
        train = pd.DataFrame({'city': ['chicago', 'los angeles']})
        test = pd.DataFrame({'city': ['chicago', 'denver']})
        y = pd.Series([1, 0])

        # HashingEncoder supports new values by design -> excluded
        for encoder_name in set(encoders.__all__) - { 'HashingEncoder' }:
            with self.subTest(encoder_name=encoder_name):
                enc = getattr(encoders, encoder_name)(handle_unknown='value')
                enc.fit(train, y)
                result = enc.transform(test)
                self.assertFalse(result.iloc[1, :].isna().all())

    def test_sklearn_compliance(self):
        """Test that the encoders are sklearn compliant."""
        for encoder_name in encoders.__all__:
            with self.subTest(encoder_name=encoder_name):
                encoder = getattr(encoders, encoder_name)()
                check_transformer_general(encoder_name, encoder)
                check_transformers_unfitted(encoder_name, encoder)
                check_n_features_in(encoder_name, encoder)
                train = pd.DataFrame({'city': ['chicago', 'los angeles']})
                y = pd.Series([1, 0])
                encoder.fit(train, y)
                self.assertTrue(hasattr(encoder, 'feature_names_out_'))
                self.assertListEqual(encoder.feature_names_in_, ['city'])
                self.assertEqual(encoder.n_features_in_, 1)
                self.assertIsInstance(encoder.get_feature_names_out(), np.ndarray)
                self.assertIsInstance(encoder.get_feature_names_in(), np.ndarray)

    def test_inverse_transform(self):
        """Test that the inverse transform works.

        We do not allow None in these data (but "none" column without any missing value is ok).
        """
        X = th.create_dataset(n_rows=100, has_missing=False)
        X_t = th.create_dataset(n_rows=50, has_missing=False)
        cols = ['underscore', 'none', 'categorical', 'categorical_int']

        for encoder_name in ['BaseNEncoder', 'BinaryEncoder', 'OneHotEncoder', 'OrdinalEncoder']:
            with self.subTest(encoder_name=encoder_name):
                # simple run
                enc = getattr(encoders, encoder_name)(verbose=1, cols=cols)
                enc.fit(X)
                th.verify_inverse_transform(X_t, enc.inverse_transform(enc.transform(X_t)))

    def test_inverse_uninitialized(self):
        """Raise an error when we call inverse_transform() before the encoder is fitted."""
        # @ToDo parametrize
        for encoder_name in {'BaseNEncoder', 'BinaryEncoder', 'OrdinalEncoder', 'OneHotEncoder'}:
            with self.subTest(encoder_name=encoder_name):
                enc = getattr(encoders, encoder_name)()
                self.assertRaises(ValueError, enc.inverse_transform, X)

    def test_inverse_wrong_feature_count(self):
        """Test that the inverse transform raises an error if the feature count is wrong."""
        x1 = [['A', 'B', 'C'], ['D', 'E', 'F'], ['G', 'H', 'I']]
        x2 = [['A', 'B'], ['C', 'D']]
        # @ToDo parametrize
        for encoder_name in {'BaseNEncoder', 'BinaryEncoder', 'OrdinalEncoder', 'OneHotEncoder'}:
            with self.subTest(encoder_name=encoder_name):
                enc = getattr(encoders, encoder_name)()
                enc.fit(x1)
                self.assertRaises(ValueError, enc.inverse_transform, x2)

    def test_inverse_wrong_feature_count_drop_invariant(self):
        """Test that the inverse transform works with drop_invariant=True."""
        x1 = [['A', 'B', 'C'], ['D', 'E', 'F'], ['G', 'H', 'I']]
        x2 = [['A', 'B'], ['C', 'D']]
        # @ToDo parametrize
        for encoder_name in {'BaseNEncoder', 'BinaryEncoder', 'OrdinalEncoder', 'OneHotEncoder'}:
            with self.subTest(encoder_name=encoder_name):
                enc = getattr(encoders, encoder_name)(drop_invariant=True)
                enc.fit(x1)
                self.assertRaises(ValueError, enc.inverse_transform, x2)

    def test_inverse_numeric(self):
        """Test that the inverse transform works with numeric data."""
        x = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        y = [0, 0, 1]

        # @ToDo parametrize
        for encoder_name in {'BaseNEncoder', 'BinaryEncoder', 'OrdinalEncoder', 'OneHotEncoder'}:
            with self.subTest(encoder_name=encoder_name):
                enc = getattr(encoders, encoder_name)()
                transformed = enc.fit_transform(x, y)
                result = enc.inverse_transform(transformed)
                self.assertTrue((x == result.to_numpy()).all())

    def test_inverse_numpy(self):
        """Test that the inverse transform works with numpy arrays.

        See issue #196
        @ToDo parametrize
        """
        for encoder_name in {'BaseNEncoder', 'BinaryEncoder', 'OrdinalEncoder', 'OneHotEncoder'}:
            with self.subTest(encoder_name=encoder_name):
                arr = np.array([['A'], ['B'], ['B'], ['C']])
                enc = getattr(encoders, encoder_name)(return_df=False)

                enc.fit(arr)
                arr_encoded = enc.transform(arr)
                arr_decoded = enc.inverse_transform(arr_encoded)

                assert np.array_equal(arr, arr_decoded)

    def test_types(self):
        """Test that the encoder can handle different data types."""
        X = pd.DataFrame(
            {
                'Int': [1, 2, 1, 2],
                'Float': [1.1, 2.2, 3.3, 4.4],
                'Complex': [3.45j, 3.45j, 3.45j, 3.45j],
                'None': [None, None, None, None],
                'Str': ['a', 'c', 'c', 'd'],
                'PdTimestamp': [
                    pd.Timestamp('2012-05-01'),
                    pd.Timestamp('2012-05-02'),
                    pd.Timestamp('2012-05-03'),
                    pd.Timestamp('2012-05-06'),
                ],
                'PdTimedelta': [
                    pd.Timedelta('1 days'),
                    pd.Timedelta('2 days'),
                    pd.Timedelta('1 days'),
                    pd.Timedelta('1 days'),
                ],
                'TimeDelta': [timedelta(-9999), timedelta(-9), timedelta(-1), timedelta(999)],
                'Bool': [False, True, True, False],
                'Tuple': [('a', 'tuple'), ('a', 'tuple'), ('a', 'tuple'), ('b', 'tuple')],
                'Categorical': pd.Categorical(
                    list('bbea'), categories=['e', 'a', 'b'], ordered=True
                ),
                # 'List': [[1,2], [2,3], [3,4], [4,5]],
                # 'Dictionary': [{1: "a", 2: "b"}, {1: "a", 2: "b"},
                #                {1: "a", 2: "b"}, {1: "a", 2: "b"}],
                # 'Set': [{'John', 'Jane'}, {'John', 'Jane'}, {'John', 'Jane'}, {'John', 'Jane'}],
                # 'Array': [array('i'), array('i'), array('i'), array('i')]
            }
        )
        y = [1, 0, 0, 1]

        for encoder_name in encoders.__all__:
            encoder = getattr(encoders, encoder_name)()
            encoder.fit_transform(X, y)

    def test_preserve_column_order(self):
        """Test that the encoder preserves the column order."""
        binary_cat_example = pd.DataFrame(
            {
                'Trend': ['UP', 'UP', 'DOWN', 'FLAT', 'DOWN', 'UP', 'DOWN', 'FLAT', 'FLAT', 'FLAT'],
                'target': [1, 1, 0, 0, 1, 0, 0, 0, 1, 1],
            },
            columns=['Trend', 'target'],
        )

        for encoder_name in encoders.__all__:
            with self.subTest(encoder_name=encoder_name):
                encoder = getattr(encoders, encoder_name)()
                result = encoder.fit_transform(binary_cat_example, binary_cat_example['target'])
                columns = result.columns

                self.assertTrue(
                    'target' in columns[-1],
                    "Target must be the last column as in the input. "
                    "This is a tricky test because 'y' is named 'target' as well.",
                )

    def test_tmp_column_name(self):
        """Test that the encoder can handle a temporary column name."""
        binary_cat_example = pd.DataFrame(
            {
                'Trend': ['UP', 'UP', 'DOWN', 'FLAT'],
                'Trend_tmp': ['UP', 'UP', 'DOWN', 'FLAT'],
                'target': [1, 1, 0, 0],
            },
            columns=['Trend', 'Trend_tmp', 'target'],
        )
        for encoder_name in encoders.__all__:
            enc = getattr(encoders, encoder_name)()
            if not enc._get_tags().get('supervised_encoder'):
                continue
            with self.subTest(encoder_name=encoder_name):
                _ = enc.fit_transform(binary_cat_example, binary_cat_example['target'])

    def test_preserve_names(self):
        """Test that the encoder preserves the column names."""
        binary_cat_example = pd.DataFrame(
            {
                'ignore': ['UP', 'UP', 'DOWN', 'FLAT'],
                'feature': ['UP', 'UP', 'DOWN', 'FLAT'],
                'target': [1, 1, 0, 0],
            },
            columns=['ignore', 'feature', 'target'],
        )

        for encoder_name in encoders.__all__:
            with self.subTest(encoder_name=encoder_name):
                encoder = getattr(encoders, encoder_name)(cols=['feature'])
                result = encoder.fit_transform(binary_cat_example, binary_cat_example['target'])
                columns = result.columns

                self.assertTrue(
                    'ignore' in columns, "Column 'ignore' is missing in: " + str(columns)
                )

    def test_unique_column_is_not_predictive(self):
        """Test that the unique column is not predictive of the label."""
        # @ToDo not sure how useful this test is.
        #  TargetEncoders set the value to the default if there is only
        #  one category but they probably should not. See discussion in issue 327
        test_encoders = [
            'LeaveOneOutEncoder',
            'WOEEncoder',
            'MEstimateEncoder',
            'JamesSteinEncoder',
            'CatBoostEncoder',
            'GLMMEncoder',
        ]
        for encoder_name in test_encoders:
            enc = getattr(encoders, encoder_name)()
            with self.subTest(encoder_name=encoder_name):
                result = enc.fit_transform(X[['unique_str']], y)
                self.assertTrue(
                    all(result.var() < 0.001),
                    'The unique string column must not be predictive of the label',
                )

    def test_cols(self):
        """Test cols argument with different data types, which are array-like or scalars."""
        cols_list = ['extra', 'invariant']
        cols_types = [
            cols_list,
            pd.Series(cols_list),
            np.array(cols_list),
            'extra',
            set(cols_list),
            ('extra', 'invariant'),
            pd.Categorical(cols_list, categories=cols_list),
        ]

        for encoder_name in encoders.__all__:
            for cols in cols_types:
                with self.subTest(encoder_name=encoder_name, cols=cols):
                    enc = getattr(encoders, encoder_name)(cols=cols)
                    enc.fit(X, y)
                    enc.transform(X_t)

    def test_non_contiguous_index(self):
        """Test if the encoder can handle non-contiguous index values."""
        for encoder_name in encoders.__all__:
            with self.subTest(encoder_name=encoder_name):
                enc = getattr(encoders, encoder_name)(cols=['x'])
                data = pd.DataFrame(
                    {'x': ['a', 'b', np.nan, 'd', 'e'], 'y': [1, 0, 1, 0, 1]}
                ).dropna()
                _ = enc.fit_transform(data[['x']], data['y'])

    def test_duplicate_index_value(self):
        """Test if the encoder can handle duplicate index values."""
        for encoder_name in encoders.__all__:
            with self.subTest(encoder_name=encoder_name):
                enc = getattr(encoders, encoder_name)(cols=['x'])
                data = pd.DataFrame(
                    {'x': ['a', 'b', 'c', 'd', 'e'], 'y': [1, 0, 1, 0, 1]}, index=[1, 2, 2, 3, 4]
                )
                result = enc.fit_transform(data[['x']], data['y'])
                self.assertEqual(5, len(result))

    def test_string_index(self):
        """Test if the encoder can handle string indices."""
        train = pd.DataFrame({'city': ['chicago', 'denver']})
        target = [0, 1]
        train.index = train.index.astype(str)

        for encoder_name in encoders.__all__:
            with self.subTest(encoder_name=encoder_name):
                enc = getattr(encoders, encoder_name)()
                result = enc.fit_transform(train, target)
                self.assertFalse(
                    result.isna().any(axis=None), 'There should not be any missing value!'
                )

    def test_get_feature_names_out(self):
        """Should return correct column names."""
        for encoder_name in encoders.__all__:
            with self.subTest(encoder_name=encoder_name):
                enc = getattr(encoders, encoder_name)()
                # Target encoders also need y
                if enc._get_tags().get('supervised_encoder'):
                    obtained = enc.fit(X, y).get_feature_names_out()
                    expected = np.array(enc.transform(X, y).columns)
                else:
                    obtained = enc.fit(X).get_feature_names_out()
                    expected = np.array(enc.transform(X).columns)
                assert_array_equal(obtained, expected)

    def test_get_feature_names_out_drop_invariant(self):
        """Should return correct column names when dropping invariant columns."""
        # TODO: What could a DF look like that results in constant
        # columns for all encoders?
        for encoder_name in encoders.__all__:
            with self.subTest(encoder_name=encoder_name):
                enc = getattr(encoders, encoder_name)(drop_invariant=True)
                # Target encoders also need y
                if enc._get_tags().get('supervised_encoder'):
                    obtained = enc.fit(X, y).get_feature_names_out()
                    expected = np.array(enc.transform(X, y).columns)
                else:
                    obtained = enc.fit(X).get_feature_names_out()
                    expected = np.array(enc.transform(X).columns)
                assert_array_equal(obtained, expected)

    def test_get_feature_names_out_not_set(self):
        """Test if get_feature_names_out() raises an error if the encoder is not fitted."""
        for encoder_name in encoders.__all__:
            with self.subTest(encoder_name=encoder_name):
                enc = getattr(encoders, encoder_name)()
                self.assertRaises(ValueError, enc.get_feature_names_out)

    def test_get_feature_names_out_after_transform(self):
        """Test if get_feature_names_out() returns the correct column names after transform."""
        for encoder_name in encoders.__all__:
            with self.subTest(encoder_name=encoder_name):
                enc = getattr(encoders, encoder_name)()
                enc.fit(X, y)
                out = enc.transform(X_t)
                self.assertEqual(set(enc.get_feature_names_out()), set(out.columns))

    def test_truncated_index(self):
        """Test if an encoder can be trained on the slice of a dataframe.

        see: https://github.com/scikit-learn-contrib/categorical-encoding/issues/152
        """
        data = pd.DataFrame(data={'x': ['A', 'B', 'C', 'A', 'B'], 'y': [1, 0, 1, 0, 1]})
        data = data.iloc[2:5]
        data2 = pd.DataFrame(data={'x': ['C', 'A', 'B'], 'y': [1, 0, 1]})
        for encoder_name in set(encoders.__all__) - {'HashingEncoder'}:
            with self.subTest(encoder_name=encoder_name):
                enc = getattr(encoders, encoder_name)()
                result = enc.fit_transform(data.x, data.y)
                enc2 = getattr(encoders, encoder_name)()
                result2 = enc2.fit_transform(data2.x, data2.y)
                self.assertTrue((result.to_numpy() == result2.to_numpy()).all())

    def test_column_transformer(self):
        """Test if the sklearn ColumnTransformer works with the encoders.

        see issue #169.
        """
        # HashingEncoder does not accept handle_missing parameter
        for encoder_name in set(encoders.__all__) - { 'HashingEncoder' }:
            with self.subTest(encoder_name=encoder_name):
                # we can only test one data type at once. Here, we test string columns.
                tested_columns = ['unique_str', 'invariant', 'underscore', 'none', 'extra']

                # ColumnTransformer instantiates the encoder twice ->
                # we have to make sure the encoder settings are correctly passed
                ct = ColumnTransformer(
                    [
                        (
                            'dummy_encoder_name',
                            getattr(encoders, encoder_name)(handle_missing='return_nan'),
                            tested_columns,
                        )
                    ]
                )
                obtained = ct.fit_transform(X, y)

                # the old-school approach
                enc = getattr(encoders, encoder_name)(handle_missing='return_nan', return_df=False)
                expected = enc.fit_transform(X[tested_columns], y)

                np.testing.assert_array_equal(obtained, expected)

    def test_error_messages(self):
        """Test if the error messages are meaningful.

        Case 1: The count of features changes must be the same in training and scoring.
        Case 2: supervised encoders must obtain 'y' of the same length as 'x' during training.
        """
        # Case 1
        data = pd.DataFrame(data={'x': ['A', 'B', 'C', 'A', 'B'], 'y': [1, 0, 1, 0, 1]})
        data2 = pd.DataFrame(data={'x': ['C', 'A', 'B'], 'x2': ['C', 'A', 'B']})
        for encoder_name in encoders.__all__:
            with self.subTest(encoder_name=encoder_name):
                enc = getattr(encoders, encoder_name)()
                enc.fit(data.x, data.y)
                self.assertRaises(ValueError, enc.transform, data2)

        # Case 2
        x = ['A', 'B', 'C']
        y_good = pd.Series([1, 0, 1])
        y_bad = pd.Series([1, 0, 1, 0])
        for encoder_name in encoders.__all__:
            enc = getattr(encoders, encoder_name)()
            if not enc._get_tags().get('supervised_encoder'):
                continue
            with self.subTest(encoder_name=encoder_name):
                self.assertRaises(ValueError, enc.fit, x, y_bad)

            with self.subTest(encoder_name=encoder_name):
                enc.fit(x, y_good)
                self.assertRaises(ValueError, enc.transform, x, y_bad)

    def test_drop_invariant(self):
        """Should drop invariant columns when drop_invariant=True."""
        x = pd.DataFrame(
            [['A', 'B', 'C'], ['A', 'B', 'C'], ['A', 'B', 'C'], ['D', 'E', 'C'], ['A', 'B', 'C']]
        )
        y = [0, 0, 1, 1, 1]

        # CatBoost does not generally deliver a constant column when the feature is constant
        # ContrastCoding schemes will always ignore invariant columns, even if set to false
        encoders_to_ignore = {
            'CatBoostEncoder', 'PolynomialEncoder', 'SumEncoder',
            'BackwardDifferenceEncoder', 'HelmertEncoder'
        }
        for encoder_name in set(encoders.__all__) - encoders_to_ignore:
            with self.subTest(encoder_name=encoder_name):
                enc1 = getattr(encoders, encoder_name)(drop_invariant=False)
                enc2 = getattr(encoders, encoder_name)(drop_invariant=True)

                result1 = enc1.fit_transform(x, y)
                result2 = enc2.fit_transform(x, y)

                self.assertTrue(len(result1.columns) > len(result2.columns))

    def test_target_encoders(self):
        """Should raise an error when the target is not provided for supervised encoders.

        See issue #206
        """
        for encoder_name in encoders.__all__:
            enc = getattr(encoders, encoder_name)()
            if not enc._get_tags().get('supervised_encoder'):
                continue
            with self.subTest(encoder_name=encoder_name):
                enc = getattr(encoders, encoder_name)(return_df=False)
                # an attempt to fit_transform() a supervised encoder without the target should
                # result into a meaningful error message
                self.assertRaises(TypeError, enc.fit_transform, X)

    def test_missing_values(self):
        """Should by default treat missing values as another valid value."""
        x_placeholder = pd.Series(['a', 'b', 'b', 'c', 'c'])
        x_nan = pd.Series(['a', 'b', 'b', np.nan, np.nan])
        x_float = pd.DataFrame({'col1': [1.0, 2.0, 2.0, np.nan, np.nan]})
        y = [0, 1, 1, 1, 1]

        for encoder_name in set(encoders.__all__) - {
            'HashingEncoder'
        }:  # HashingEncoder currently violates it
            with self.subTest(encoder_name=encoder_name):
                enc = getattr(encoders, encoder_name)()
                result_placeholder = enc.fit_transform(x_placeholder, y)

                enc = getattr(encoders, encoder_name)()
                result_nan = enc.fit_transform(x_nan, y)

                enc = getattr(encoders, encoder_name)(cols='col1')
                result_float = enc.fit_transform(x_float, y)

                pd.testing.assert_frame_equal(result_placeholder, result_nan)
                np.testing.assert_equal(result_placeholder.values, result_float.values)

    def test_metamorphic(self):
        """Test the metamorphic property of the encoders.

        This means that the output should remain unchanged when we slightly alter the input data
        (e.g. other labels) or an irrelevant argument.

        We include the following cases:
        - Baseline
        - Different strings, but with the same alphabetic ordering
        - Input as DataFrame
        - Input as Series with category data type
        - Input as Numpy
        - Different strings, reversed alphabetic ordering (it works because we look at
          the order of appearance, not at alphabetic order)

        Note that the hashing encoder is not expected to be metamorphic.
        """
        x1 = ['A', 'B', 'B']
        x2 = [
            'Apple',
            'Banana',
            'Banana',
        ]
        x3 = pd.DataFrame(data={'x': ['A', 'B', 'B']})
        x4 = pd.Series(['A', 'B', 'B'], dtype='category')
        x5 = np.array(['A', 'B', 'B'])
        x6 = [
            'Z',
            'Y',
            'Y',
        ]

        y = [1, 1, 0]

        for encoder_name in set(encoders.__all__) - {
            'HashingEncoder'
        }:  # Hashing encoder is, by definition, not invariant to data changes
            with self.subTest(encoder_name=encoder_name):
                enc1 = getattr(encoders, encoder_name)()
                result1 = enc1.fit_transform(x1, y)

                enc2 = getattr(encoders, encoder_name)()
                result2 = enc2.fit_transform(x2, y)
                self.assertTrue(result1.equals(result2))

                enc3 = getattr(encoders, encoder_name)()
                result3 = enc3.fit_transform(x3, y)
                self.assertTrue((result1.to_numpy() == result3.to_numpy()).all())

                enc4 = getattr(encoders, encoder_name)()
                result4 = enc4.fit_transform(x4, y)
                self.assertTrue(result1.equals(result4))

                enc5 = getattr(encoders, encoder_name)()
                result5 = enc5.fit_transform(x5, y)
                self.assertTrue(result1.equals(result5))

                # gray encoder actually does re-order inputs
                # rankhot encoder respects order, in this example the order is switched
                if encoder_name not in ['GrayEncoder', 'RankHotEncoder']:
                    enc6 = getattr(encoders, encoder_name)()
                    result6 = enc6.fit_transform(x6, y)
                    self.assertTrue(result1.equals(result6))

                # Arguments
                enc9 = getattr(encoders, encoder_name)(return_df=False)
                result9 = enc9.fit_transform(x1, y)
                self.assertTrue((result1.to_numpy() == result9).all())

                enc10 = getattr(encoders, encoder_name)(verbose=True)
                result10 = enc10.fit_transform(x1, y)
                self.assertTrue(result1.equals(result10))

                # Note: If the encoder does not support these arguments/argument values,
                # it is OK/expected to fail.
                # Note: The indicator approach is not tested because it adds columns -> the
                # encoders that support it are expected to fail.

                # enc11 = getattr(encoders, encoder_name)(handle_unknown='return_nan',
                #         handle_missing='return_nan')
                # Quite a few algorithms fail here because of handle_missing
                # result11 = enc11.fit_transform(x1, y)
                # self.assertTrue((result1.values == result11.values).all(),
                # 'The data do not contain any missing or new value -> the result should
                # be unchanged.')

                enc12 = getattr(encoders, encoder_name)(
                    handle_unknown='value', handle_missing='value'
                )
                result12 = enc12.fit_transform(x1, y)
                self.assertTrue(
                    result1.equals(result12),
                    'The data do not contain any missing or new value -> '
                    'the result should be unchanged.',
                )

                # enc13 = getattr(encoders, encoder_name)(handle_unknown='error',
                # handle_missing='error', cols=['x'])
                # Quite a few algorithms fail here because of handle_missing
                # result13 = enc13.fit_transform(x3, y)
                # self.assertTrue((result1.values == result13.values).all(),
                # 'The data do not contain any missing or new value ->
                # the result should be unchanged.')

    def test_pandas_index(self):
        """Should work with pandas index.

        See https://github.com/scikit-learn-contrib/categorical-encoding/pull/224
        """
        df = pd.DataFrame(
            {'hello': ['a', 'b', 'c'], 'world': [0, 1, 0]}, columns=pd.Index(['hello', 'world'])
        )
        cols = df.select_dtypes(include='object').columns

        for encoder_name in set(encoders.__all__) - {'HashingEncoder'}:
            with self.subTest(encoder_name=encoder_name):
                enc = getattr(encoders, encoder_name)(cols=cols)
                enc.fit_transform(df, df['world'])

    def test_mismatched_indexes(self):
        """Should work with mismatched indexes."""
        df = pd.DataFrame({'x': ['a', 'b', 'b']}, index=[7, 5, 9])
        y_list = [1, 0, 1]
        for encoder_name in encoders.__all__:
            with self.subTest(encoder_name=encoder_name):
                enc = getattr(encoders, encoder_name)()
                out = enc.fit_transform(df, y_list)
                self.assertFalse(out.isna().any().any())

    def test_numbers_as_strings_with_numpy_output(self):
        """Should work with numbers as strings.

        See issue #229.
        """
        X = np.array(['11', '12', '13', '14', '15'])
        oe = encoders.OrdinalEncoder(return_df=False)
        oe.fit(X)

    def test_columns(self):
        """Should convert only selected columns.

        If no selection is made all columns of type object should be converted.
        """
        # Convert only selected columns. Leave the remaining string columns untouched.
        oe = encoders.OrdinalEncoder(cols=['underscore'])
        result = oe.fit_transform(X)
        self.assertTrue(result['underscore'].min() == 1, 'should newly be a number')
        self.assertTrue(result['unique_str'].min() == '0', 'should still be a string')

        # If no selection is made, convert all (and only) object columns
        oe = encoders.OrdinalEncoder()
        result = oe.fit_transform(X)
        self.assertTrue(result['unique_str'].min() == 1, 'should newly be a number')
        self.assertTrue(result['invariant'].min() == 1, 'should newly be a number')
        self.assertTrue(result['underscore'].min() == 1, 'should newly be a number')
        self.assertTrue(result['none'].min() == 1, 'should newly be a number')
        self.assertTrue(result['extra'].min() == 1, 'should newly be a number')
        self.assertTrue(result['categorical'].min() == 1, 'should newly be a number')
        self.assertTrue(result['na_categorical'].min() == 1, 'should newly be a number')
        self.assertTrue(result['categorical_int'].min() == 1, 'should newly be a number')

        self.assertTrue(result['float'].min() < 1, 'should still be a number and untouched')
        self.assertTrue(result['float_edge'].min() < 1, 'should still be a number and untouched')
        self.assertTrue(result['unique_int'].min() < 1, 'should still be a number and untouched')

    def test_ignored_columns_are_untouched(self):
        """Should not change None values of ignored columns.

        See: https://github.com/scikit-learn-contrib/category_encoders/pull/261
        """
        X = pd.DataFrame({'col1': ['A', 'B', None], 'col2': ['C', 'D', None]})
        y = [1, 0, 1]

        for encoder_name in set(encoders.__all__):
            with self.subTest(encoder_name=encoder_name):
                enc = getattr(encoders, encoder_name)(cols=['col1'])
                out = enc.fit_transform(X, y)
                self.assertTrue(out.col2[2] is None)
