"""Unit tests for the CountTargetEncoder."""

from unittest import TestCase

import category_encoders as encoders
import numpy as np
import pandas as pd
from scipy.special import expit

import tests.helpers as th

np_X = th.create_array(n_rows=100)
np_X_t = th.create_array(n_rows=50, extras=True)
np_y = np.random.randn(np_X.shape[0]) > 0.5
np_y_t = np.random.randn(np_X_t.shape[0]) > 0.5
X = th.create_dataset(n_rows=100)
X_t = th.create_dataset(n_rows=50, extras=True)
y = pd.DataFrame(np_y)
y_t = pd.DataFrame(np_y_t)


def expected_binary_log_odds(counts_pos, counts_neg, min_samples_leaf, smoothing, prior_pos):
    """Independently derive the expected binary log-odds from the documented formula."""
    n = counts_pos + counts_neg
    prior_neg = 1 - prior_pos
    weight = expit((n - min_samples_leaf) / smoothing)
    p_pos = weight * (counts_pos / n) + (1 - weight) * prior_pos
    p_neg = weight * (counts_neg / n) + (1 - weight) * prior_neg
    return np.log(p_pos / p_neg) - np.log(prior_pos / prior_neg)


class TestCountTargetEncoder(TestCase):
    """Unit tests for the CountTargetEncoder."""

    def test_binary_single_column(self):
        """Binary targets emit a single output column that replaces the input."""
        train = pd.DataFrame({'city': ['chicago', 'chicago', 'denver', 'denver', 'denver']})
        target = [1, 0, 1, 1, 0]

        enc = encoders.CountTargetEncoder()
        result = enc.fit_transform(train, target)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertListEqual(list(result.columns), ['city'])
        th.verify_numeric(result)

    def test_binary_expected_values(self):
        """The binary column matches the documented smoothed log-odds formula."""
        train = pd.DataFrame({'city': ['chicago', 'chicago', 'denver', 'denver', 'denver']})
        target = pd.Series([1, 0, 1, 1, 0])
        enc = encoders.CountTargetEncoder(min_samples_leaf=20, smoothing=10)
        result = enc.fit_transform(train, target)

        expected_chicago = expected_binary_log_odds(
            counts_pos=1, counts_neg=1, min_samples_leaf=20, smoothing=10, prior_pos=3 / 5
        )
        expected_denver = expected_binary_log_odds(
            counts_pos=2, counts_neg=1, min_samples_leaf=20, smoothing=10, prior_pos=3 / 5
        )
        self.assertAlmostEqual(result.loc[0, 'city'], expected_chicago)
        self.assertAlmostEqual(result.loc[2, 'city'], expected_denver)

    def test_binary_bool_and_int_target_are_equivalent(self):
        """Bool, int and label-encoded string targets encode identically."""
        train = pd.DataFrame({'city': ['chicago', 'denver'] * 6})
        result_int = encoders.CountTargetEncoder().fit_transform(train, [1, 0] * 6)
        result_bool = encoders.CountTargetEncoder().fit_transform(train, [True, False] * 6)
        result_str = encoders.CountTargetEncoder().fit_transform(train, ['yes', 'no'] * 6)

        self.assertTrue(result_int.equals(result_bool))
        pd.testing.assert_frame_equal(result_int, result_str)

    def test_smoothing_pulls_small_categories_toward_zero(self):
        """Strong regularization shrinks category evidence toward zero."""
        train = pd.DataFrame({'city': ['chicago'] * 4 + ['denver'] * 4})
        target = [1, 1, 1, 1, 0, 0, 0, 0]

        # min_samples_leaf far above every category size -> the prior dominates
        enc = encoders.CountTargetEncoder(min_samples_leaf=10**9)
        result = enc.fit_transform(train, target)
        self.assertTrue(np.allclose(result['city'], 0.0))

        # large category counts relative to the S-curve -> close to the raw log-odds ratio
        enc = encoders.CountTargetEncoder(min_samples_leaf=1, smoothing=1)
        result = enc.fit_transform(train, target)
        self.assertAlmostEqual(result.loc[0, 'city'], expected_binary_log_odds(4, 0, 1, 1, 0.5))
        self.assertAlmostEqual(result.loc[4, 'city'], expected_binary_log_odds(0, 4, 1, 1, 0.5))

    def test_smoothing_must_be_positive(self):
        """A non-positive smoothing value is rejected."""
        train = pd.DataFrame({'city': ['chicago', 'denver']})
        with self.assertRaises(ValueError):
            encoders.CountTargetEncoder(smoothing=0).fit(train, [1, 0])

    def test_counts_stored_per_class(self):
        """Fit stores the per-class counts indexed by the original categories."""
        train = pd.DataFrame({'c1': ['a', 'a', 'b', 'b', 'c', 'c', 'c']})
        target = [0, 1, 2, 0, 1, 2, 1]

        enc = encoders.CountTargetEncoder(cols=['c1']).fit(train, target)
        counts = enc.counts_['c1']

        self.assertEqual(counts.loc['a', 0], 1)
        self.assertEqual(counts.loc['a', 1], 1)
        self.assertEqual(counts.loc['c', 1], 2)
        self.assertEqual(counts.loc['c', 2], 1)
        self.assertEqual(counts.to_numpy().sum(), len(target))

    def test_multiclass_shapes(self):
        """Multiclass targets emit one column per class, keeping the other columns."""
        train = pd.DataFrame(
            {'c1': ['a', 'a', 'b', 'b', 'c', 'c', 'c'], 'num': [1, 2, 3, 4, 5, 6, 7]}
        )
        target = [0, 1, 2, 0, 1, 2, 1]

        enc = encoders.CountTargetEncoder(cols=['c1'])
        result = enc.fit_transform(train, target)

        self.assertEqual(result.shape, (7, 4))
        self.assertListEqual(list(result.columns), ['c1_0', 'c1_1', 'c1_2', 'num'])
        th.verify_numeric(result)
        self.assertListEqual(list(enc.get_feature_names_out()), list(result.columns))

    def test_multiclass_column_order(self):
        """Expanded columns stay at the position of the original column."""
        train = pd.DataFrame({'first': ['x', 'y', 'z'], 'c1': ['a', 'b', 'c'], 'last': [1, 2, 3]})
        target = [0, 1, 2]

        result = encoders.CountTargetEncoder(cols=['c1']).fit_transform(train, target)

        self.assertListEqual(list(result.columns), ['first', 'c1_0', 'c1_1', 'c1_2', 'last'])

    def test_multiclass_string_labels(self):
        """Column names use the original class labels of string targets."""
        train = pd.DataFrame({'c1': ['a', 'a', 'b', 'b', 'c', 'c', 'c']})
        target = ['red', 'green', 'blue', 'red', 'green', 'blue', 'red']

        result = encoders.CountTargetEncoder(cols=['c1']).fit_transform(train, target)

        self.assertListEqual(list(result.columns), ['c1_blue', 'c1_green', 'c1_red'])

    def test_multiclass_expected_values(self):
        """Each multiclass column holds the log-evidence against the prior."""
        train = pd.DataFrame({'c1': ['a', 'a', 'b', 'b', 'c', 'c', 'c']})
        target = [0, 1, 2, 0, 1, 2, 1]
        enc = encoders.CountTargetEncoder(cols=['c1'], min_samples_leaf=20, smoothing=10)
        result = enc.fit_transform(train, target)

        prior = np.array([2 / 7, 3 / 7, 2 / 7])
        for _category, counts, n, row in [
            ('a', [1, 1, 0], 2, 0),
            ('b', [1, 0, 1], 2, 2),
            ('c', [0, 2, 1], 3, 4),
        ]:
            weight = expit((n - 20) / 10)
            smoothed = weight * np.array(counts) / n + (1 - weight) * prior
            expected = np.log(smoothed / prior)
            self.assertTrue(np.allclose(result.loc[row, ['c1_0', 'c1_1', 'c1_2']], expected))

    def test_multiclass_drop_invariant(self):
        """Invariant columns are dropped in the multiclass expansion too."""
        train = pd.DataFrame({'constant': ['A'] * 4, 'varying': ['a', 'b', 'c', 'a']})
        target = [0, 1, 2, 0]

        enc = encoders.CountTargetEncoder(cols=['constant', 'varying'], drop_invariant=True)
        result = enc.fit_transform(train, target)

        self.assertNotIn('constant_0', result.columns)
        self.assertIn('varying_0', result.columns)

    def test_unknown_value(self):
        """Unknown categories map to zero evidence by default."""
        train = pd.DataFrame({'city': ['chicago', 'denver']})
        test = pd.DataFrame({'city': ['chicago', 'austin']})
        target = [1, 0]

        result = encoders.CountTargetEncoder().fit(train, target).transform(test)

        self.assertEqual(result.loc[1, 'city'], 0.0)

    def test_unknown_return_nan(self):
        """Unknown categories map to NaN with handle_unknown='return_nan'."""
        train = pd.DataFrame({'city': ['chicago', 'denver']})
        test = pd.DataFrame({'city': ['chicago', 'austin']})
        target = [1, 0]

        enc = encoders.CountTargetEncoder(handle_unknown='return_nan')
        result = enc.fit(train, target).transform(test)

        self.assertTrue(pd.isna(result.loc[1, 'city']))

    def test_unknown_error(self):
        """Unknown categories raise with handle_unknown='error'."""
        train = pd.DataFrame({'city': ['chicago', 'denver']})
        test = pd.DataFrame({'city': ['chicago', 'austin']})
        target = [1, 0]

        enc = encoders.CountTargetEncoder(handle_unknown='error')
        enc.fit(train, target)
        with self.assertRaises(ValueError):
            enc.transform(test)

    def test_missing_is_a_countable_category(self):
        """By default missing values are counted like any other category."""
        x_placeholder = pd.Series(['a', 'b', 'b', 'c', 'c'])
        x_nan = pd.Series(['a', 'b', 'b', np.nan, np.nan])
        target = [0, 1, 1, 1, 1]

        result_placeholder = encoders.CountTargetEncoder().fit_transform(x_placeholder, target)
        result_nan = encoders.CountTargetEncoder().fit_transform(x_nan, target)

        pd.testing.assert_frame_equal(result_placeholder, result_nan)

    def test_missing_value_at_transform(self):
        """Missing values on clean fit data map to zero evidence by default."""
        train = pd.DataFrame({'city': ['chicago', 'denver']})
        test = pd.DataFrame({'city': ['chicago', np.nan]})
        target = [1, 0]

        result = encoders.CountTargetEncoder().fit(train, target).transform(test)

        self.assertEqual(result.loc[1, 'city'], 0.0)

    def test_missing_return_nan(self):
        """Missing values map to NaN with handle_missing='return_nan', at fit and transform time."""
        train = pd.DataFrame({'city': ['chicago', 'denver']})
        target = [1, 0]
        enc = encoders.CountTargetEncoder(handle_missing='return_nan')
        enc.fit(train, target)

        result_test = enc.transform(pd.DataFrame({'city': ['chicago', np.nan]}))
        result_train_nan = encoders.CountTargetEncoder(handle_missing='return_nan').fit_transform(
            pd.DataFrame({'city': ['chicago', np.nan]}), [1, 0]
        )

        self.assertTrue(pd.isna(result_test.loc[1, 'city']))
        self.assertTrue(pd.isna(result_train_nan.loc[1, 'city']))

    def test_missing_error(self):
        """Missing values raise with handle_missing='error'."""
        train = pd.DataFrame({'city': ['chicago', np.nan]})
        target = [1, 0]

        enc = encoders.CountTargetEncoder(handle_missing='error')
        with self.assertRaises(ValueError):
            enc.fit(train, target)

    def test_continuous_target_raises(self):
        """A continuous target raises NotImplementedError naming the binning follow-up."""
        train = pd.DataFrame({'city': ['chicago', 'denver', 'denver', 'austin']})
        target = [0.5, 1.2, 0.7, 2.1]

        enc = encoders.CountTargetEncoder()
        with self.assertRaises(NotImplementedError) as context:
            enc.fit(train, target)

        self.assertIn('binning', str(context.exception))

    def test_duplicate_index(self):
        """Duplicate index values do not corrupt the per-class counts."""
        train = pd.DataFrame(
            {'x': ['a', 'b', 'b', 'c', 'c'], 'y': [1, 0, 1, 0, 1]}, index=[1, 2, 2, 3, 4]
        )

        result = encoders.CountTargetEncoder(cols=['x']).fit_transform(train[['x']], train['y'])

        self.assertEqual(len(result), 5)
        self.assertTrue(np.isfinite(result.to_numpy()).all())

    def test_fit_transform_equals_fit_then_transform(self):
        """Transform does not depend on y, so both entry points agree."""
        train = pd.DataFrame({'city': ['chicago', 'denver', 'chicago']})
        target = [1, 0, 0]

        enc = encoders.CountTargetEncoder()
        fit_transformed = enc.fit_transform(train, target)
        enc2 = encoders.CountTargetEncoder()
        transformed = enc2.fit(train, target).transform(train)

        pd.testing.assert_frame_equal(fit_transformed, transformed)

    def test_multiclass_numpy_input(self):
        """The encoder works on numpy input and binary/multiclass targets."""
        enc = encoders.CountTargetEncoder()
        enc.fit(np_X, np_y)
        th.verify_numeric(enc.transform(np_X_t))
        th.verify_numeric(enc.transform(np_X_t, np_y_t))

    def test_string_target_multiclass(self):
        """A 3-class string target expands to three numeric columns."""
        train = pd.DataFrame({'city': ['a', 'b', 'c'] * 3})
        target = pd.Series(['yes', 'no', 'maybe'] * 3)

        result = encoders.CountTargetEncoder().fit_transform(train, target)

        self.assertListEqual(list(result.columns), ['city_maybe', 'city_no', 'city_yes'])
        th.verify_numeric(result)
