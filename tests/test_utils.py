"""Tests for the utils module."""
from unittest import TestCase  # or `from unittest import ...` if on Python 3.4+

import category_encoders as encoders
import numpy as np
import pandas as pd
import pytest
from category_encoders.utils import (
    BaseEncoder,
    build_min_group_map,
    convert_input_vector,
    convert_inputs,
    flatten_reverse_dict,
    get_categorical_cols,
    get_generated_cols,
)
from packaging.version import Version
from sklearn import __version__ as skl_version
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError


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

    def test_determine_fit_columns_use_all_cols(self):
        """Test that _determine_fit_columns with use_all_cols=True returns all columns."""
        df = pd.DataFrame({'str_col': ['a', 'b'], 'int_col': [1, 2], 'float_col': [1.0, 2.0]})

        class DummyEncoder(BaseEncoder, BaseEstimator, TransformerMixin):
            def _fit(self, X, y=None):
                return self

            def transform(self, X, y=None, override_return_df=False):
                return X

        enc = DummyEncoder(cols='all')
        self.assertTrue(enc.use_all_cols)
        self.assertFalse(enc.use_default_cols)

        enc.fit(df)
        self.assertEqual(sorted(enc.cols), sorted(df.columns.tolist()))

        # Refit with different columns should re-detect all
        df2 = pd.DataFrame({'a': ['x', 'y'], 'b': [10, 20]})
        enc.fit(df2)
        self.assertEqual(sorted(enc.cols), sorted(df2.columns.tolist()))

    @pytest.mark.skipif(Version(skl_version) < Version('1.2'), reason='requires sklearn > 1.2')
    def test_sklearn_pandas_out_refit(self):
        """Test that the encoder can be refit with sklearn and pandas."""
        # Thanks to Issue#437
        df = pd.DataFrame({'C1': ['a', 'a'], 'C2': ['c', 'd']})
        self.encoder.set_output(transform='pandas')
        self.encoder.fit_transform(df.iloc[:1])
        out = self.encoder.fit_transform(df.rename(columns={'C1': 'X1', 'C2': 'X2'}))
        self.assertTrue(list(out.columns) == ['X1', 'X2'])

    @pytest.mark.skipif(Version(skl_version) < Version('1.2'), reason='requires sklearn > 1.2')
    def test_global_transform_output_pandas(self):
        """Encoders must fit under a global transform_output='pandas' config (Issue#488)."""
        import category_encoders as encoders
        from sklearn import config_context
        from sklearn.compose import ColumnTransformer

        X = pd.DataFrame({'color': ['red', 'blue', 'green', 'red']})
        y = [1, 0, 1, 0]
        with config_context(transform_output='pandas'):
            out = encoders.OrdinalEncoder().fit_transform(X, y)
            self.assertIsInstance(out, pd.DataFrame)

            ct = ColumnTransformer(
                [('enc', encoders.TargetEncoder(cols=['color']), ['color'])],
                remainder='passthrough',
            )
            ct.set_output(transform='pandas')
            ct_out = ct.fit_transform(X, y)
            self.assertIsInstance(ct_out, pd.DataFrame)


class TestNdarrayTransform(TestCase):
    """Arraylike input at transform: positional name re-attachment and strictness."""

    def test_transform_ndarray_before_fit_raises_not_fitted(self):
        """An unfitted encoder raises NotFittedError for arraylike input."""
        with self.assertRaises(NotFittedError):
            encoders.OrdinalEncoder().transform(np.array([['a'], ['b']]))

    def test_transform_ndarray_wrong_width_raises(self):
        """Arraylike width stays strict: a mismatched ndarray raises the dimension error."""
        df = pd.DataFrame({'str_col': ['a', 'b', 'c']})
        enc = encoders.OrdinalEncoder().fit(df)
        with self.assertRaisesRegex(ValueError, 'Unexpected input dimension'):
            enc.transform(np.array([['a', 'b'], ['b', 'c']]))

    def test_transform_ndarray_name_stable_when_fitted_on_ndarray(self):
        """An encoder fitted on an ndarray re-attaches its positional integer names."""
        X = np.array([['a', 1.0], ['b', 2.0], ['c', 3.0], ['a', 4.0]])
        enc = encoders.OrdinalEncoder()
        enc.fit(X)
        self.assertEqual(enc.feature_names_in_, [0, 1])
        out_first = enc.transform(X)
        out_second = enc.transform(X.copy())
        np.testing.assert_array_equal(out_first.to_numpy(), out_second.to_numpy())

    def test_transform_superset_dataframe_passes_extra_columns_through(self):
        """Extra DataFrame columns pass through next to the encoded ones (GH #355, GH #367)."""
        df = pd.DataFrame({'str_col': ['a', 'b', 'c', 'a'], 'num_col': [1.0, 2.0, 3.0, 4.0]})
        wide = df.copy()
        wide['extra'] = ['p', 'q', 'r', 's']
        enc = encoders.OrdinalEncoder()
        expected = enc.fit(df).transform(df)
        out = enc.transform(wide)
        self.assertIn('extra', out.columns)
        np.testing.assert_array_equal(out['str_col'].to_numpy(), expected['str_col'].to_numpy())
        np.testing.assert_array_equal(out['num_col'].to_numpy(), wide['num_col'].to_numpy())

    def test_transform_missing_encoded_column_raises_clear_error(self):
        """A frame without the encoded columns raises an error naming them."""
        df = pd.DataFrame({'str_col': ['a', 'b', 'c', 'a'], 'num_col': [1.0, 2.0, 3.0, 4.0]})
        enc = encoders.OrdinalEncoder(cols=['str_col']).fit(df)
        renamed = df.rename(columns={'str_col': 'other'})
        with self.assertRaisesRegex(ValueError, 'str_col'):
            enc.transform(renamed)

    def test_fit_ndarray_with_named_cols_error_lists_remediation(self):
        """Fitting an ndarray with named cols explains that names cannot be recovered."""
        X = np.array([['a', 1.0], ['b', 2.0], ['c', 3.0]])
        enc = encoders.OrdinalEncoder(cols=['str_col'])
        with self.assertRaisesRegex(ValueError, 'set_output'):
            enc.fit(X)


class TestUtilsMutationHardening(TestCase):
    """Targeted tests for survivors of the mutation-testing campaign in utils."""

    def test_flatten_reverse_dict_inverts_nested_mapping(self):
        """flatten_reverse_dict inverts {outer: {inner: value}} into {value: (outer, inner)}."""
        nested = {'a': {'b': 1, 'c': 2}}
        self.assertEqual({1: ('a', 'b'), 2: ('a', 'c')}, flatten_reverse_dict(nested))

    def test_get_categorical_cols_selects_string_dtyped_columns(self):
        """get_categorical_cols selects object/category/string-dtyped columns only."""
        df = pd.DataFrame(
            {
                'text': ['a', 'b'],
                'category': pd.Series(['a', 'b'], dtype='category'),
                'string': pd.Series(['a', 'b'], dtype='string'),
                'number': [1.0, 2.0],
            }
        )
        self.assertEqual(['text', 'category', 'string'], get_categorical_cols(df))

    def test_build_min_group_map_does_not_lump_when_group_sizes_sum_to_zero(self):
        """A group_sizes series summing to zero must not lump anything."""
        sizes = pd.Series([0, 0, 5], index=['a', 'b', 'c'])
        kept, lumped = build_min_group_map(sizes, 1, None, None)
        self.assertEqual({'a': 0, 'b': 0, 'c': 5}, kept.to_dict())
        self.assertEqual({}, lumped)

    def test_get_generated_cols_returns_encoded_and_to_transform_columns(self):
        """get_generated_cols unions encoded and to-transform columns."""
        X_original = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        X_transformed = pd.DataFrame({'a': [1, 2], 'a_enc': [1, 2]})
        self.assertEqual(['a', 'a_enc'], get_generated_cols(X_original, X_transformed, ['a']))
        identical = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        self.assertEqual([], get_generated_cols(X_original, identical, []))

    def test_composite_cols_validation_rejects_malformed_groups(self):
        """_validate_composite_cols rejects single-member, non-tuple, and unknown members."""
        X = pd.DataFrame({'a': ['x', 'y'], 'b': [1, 2]})
        y = pd.Series([0, 1])
        for bad_spec in [(('a',),), (['a', 'b'],), (('a', 'nope'),)]:
            with self.subTest(bad_spec=bad_spec):
                encoder = encoders.TargetEncoder(composite_cols=bad_spec)
                with self.assertRaises(ValueError):
                    encoder.fit(X, y)

    def test_composite_cols_accepts_two_distinct_groups(self):
        """Two distinct composite groups fit; a join collision is rejected.

        Kills the _validate_composite_cols length/membership/duplicate-name mutants
        (mutmut_8/-9/-19/-22/-28): valid two-group specs must fit, and two groups whose
        joined names collide must raise ValueError.
        """
        X = pd.DataFrame(
            {
                'p': ['x', 'y', 'z'],
                'q': ['1', '2', '3'],
                'r': ['u', 'v', 'w'],
                's': ['a', 'b', 'c'],
            }
        )
        y = pd.Series([1, 0, 1])
        encoder = encoders.TargetEncoder(composite_cols=[('p', 'q'), ('r', 's')])
        encoder.fit(X, y)
        self.assertIn('p|q', encoder.mapping)
        self.assertIn('r|s', encoder.mapping)

        X_piped = pd.DataFrame(
            {
                'p': ['x', 'y', 'z'],
                'q': ['1', '2', '3'],
                'r': ['u', 'v', 'w'],
                'p|q': ['a', 'b', 'c'],
                'q|r': ['d', 'e', 'f'],
            }
        )
        collision = encoders.TargetEncoder(composite_cols=[('p|q', 'r'), ('p', 'q|r')])
        with self.assertRaises(ValueError):
            collision.fit(X_piped, y)

    def test_transform_output_restores_fitted_dtypes_for_arraylike_input(self):
        """Arraylike transform input must restore the fitted object/category dtypes.

        Only object and category columns carry encoding semantics an ndarray loses
        (GH #406); kills _restore_fitted_dtypes mutmut_3/-4 (the dtype gate).
        """
        X = pd.DataFrame(
            {
                'str_col': ['a', 'b', 'c'],
                'cat_col': pd.Series(['x', 'y', 'z'], dtype='category'),
            }
        )
        encoder = encoders.OrdinalEncoder(cols=['str_col'])
        encoder.fit(X)
        out = encoder.transform(X.to_numpy())
        self.assertEqual('int64', str(out['str_col'].dtype))
        self.assertEqual('category', str(out['cat_col'].dtype))

    def test_get_feature_names_in_requires_and_reflects_fit(self):
        """get_feature_names_in raises NotFittedError before fit and reflects columns after."""
        encoder = encoders.OrdinalEncoder(cols=['str_col'])
        with self.assertRaises(NotFittedError):
            encoder.get_feature_names_in()
        X = pd.DataFrame({'str_col': ['a', 'b'], 'num': [1, 2]})
        encoder.fit(X)
        np.testing.assert_array_equal(
            np.array(['str_col', 'num']), encoder.get_feature_names_in()
        )
