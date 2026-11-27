"""Unit tests for the LeaveOneOutEncoder."""

from unittest import TestCase  # or `from unittest import ...` if on Python 3.4+

import category_encoders as encoders
import numpy as np
import pandas as pd

import tests.helpers as th


class TestLeaveOneOutEncoder(TestCase):
    """Unit tests for the LeaveOneOutEncoder."""

    def test_leave_one_out(self):
        """Test basic functionality on a diverse dataset."""
        np_X = th.create_array(n_rows=100)
        np_X_t = th.create_array(n_rows=50, extras=True)
        np_y = np.random.randn(np_X.shape[0]) > 0.5
        np_y_t = np.random.randn(np_X_t.shape[0]) > 0.5
        X = th.create_dataset(n_rows=100)
        X_t = th.create_dataset(n_rows=50, extras=True)
        y = pd.DataFrame(np_y)
        y_t = pd.DataFrame(np_y_t)
        enc = encoders.LeaveOneOutEncoder(verbose=1, sigma=0.1)
        enc.fit(X, y)
        th.verify_numeric(enc.transform(X_t))
        th.verify_numeric(enc.transform(X_t, y_t))

    def test_leave_one_out_values(self):
        """Test that the fitted values are correct."""
        df = pd.DataFrame({'color': ['a', 'a', 'a', 'b', 'b', 'b'], 'outcome': [1, 0, 0, 1, 0, 1]})

        X = df.drop('outcome', axis=1)
        y = df.drop('color', axis=1)

        ce_leave = encoders.LeaveOneOutEncoder(cols=['color'])
        obtained = ce_leave.fit_transform(X, y['outcome'])

        self.assertEqual([0.0, 0.5, 0.5, 0.5, 1.0, 0.5], list(obtained['color']))

    def test_refit(self):
        """Test that the encoder can be refit if fit is called twice with different data."""
        x_a = pd.DataFrame(data=['1', '2', '2', '2', '2', '2'], columns=['col_a'])
        x_b = pd.DataFrame(
            data=['1', '1', '1', '2', '2', '2'], columns=['col_b']
        )  # different values and name
        y_dummy = [True, False, True, False, True, False]
        encoder = encoders.LeaveOneOutEncoder()
        encoder.fit(x_a, y_dummy)
        encoder.fit(x_b, y_dummy)
        mapping = encoder.mapping
        self.assertEqual(1, len(mapping))
        self.assertIn('col_b', mapping)  # the model should have the updated mapping
        expected = pd.DataFrame(
            {'sum': [2.0, 1.0], 'count': [3, 3]}, index=['1', '2'], columns=['sum', 'count']
        )
        np.testing.assert_equal(expected.values, mapping['col_b'].values)

    def test_leave_one_out_unique(self):
        """Test that unique levels are encoded as the global mean."""
        X = pd.DataFrame(data=['1', '2', '2', '2', '3'], columns=['col'])
        y = np.array([1, 0, 1, 0, 1])

        encoder = encoders.LeaveOneOutEncoder(handle_unknown='value')
        result = encoder.fit(X, y).transform(X, y)

        self.assertFalse(result.isna().any().any(), 'There should not be any missing value')
        expected = pd.DataFrame(data=[y.mean(), 0.5, 0, 0.5, y.mean()], columns=['col'])
        pd.testing.assert_frame_equal(expected, result)

    def test_handle_missing_value_nan_in_training(self):
        """Should encode missing values with the mean in training."""
        df = pd.DataFrame(
            {'color': [np.nan, np.nan, np.nan, 'b', 'b', 'b'], 'outcome': [2, 2, 0, 1, 0, 1]}
        )

        X = df.drop('outcome', axis=1)
        y = df.drop('color', axis=1)

        ce_leave = encoders.LeaveOneOutEncoder(cols=['color'], handle_missing='value')
        obtained = ce_leave.fit_transform(X, y['outcome'])

        self.assertEqual([1, 1, 2, 0.5, 1.0, 0.5], list(obtained['color']))

    def test_handle_missing_value_nan_not_in_training(self):
        """Should encode missing values with the global mean if not present in training."""
        df = pd.DataFrame(
            {'color': ['a', 'a', 'a', 'b', 'b', 'b'], 'outcome': [1.6, 0, 0, 1, 0, 1]}
        )

        train = df.drop('outcome', axis=1)
        target = df.drop('color', axis=1)
        test = pd.Series([np.nan, 'b'], name='color')
        test_target = pd.Series([0, 0])

        ce_leave = encoders.LeaveOneOutEncoder(cols=['color'], handle_missing='value')
        ce_leave.fit(train, target['outcome'])
        obtained = ce_leave.transform(test, test_target)

        self.assertEqual([0.6, 1.0], list(obtained['color']))

    def test_handle_missing_value(self):
        """Should encode missing values with the global mean."""
        df = pd.DataFrame({'color': ['a', 'a', 'a', 'b', 'b', 'b'], 'outcome': [1, 0, 0, 1, 0, 1]})

        train = df.drop('outcome', axis=1)
        target = df.drop('color', axis=1)
        test = pd.Series([np.nan, 'b'], name='color')

        ce_leave = encoders.LeaveOneOutEncoder(cols=['color'], handle_missing='value')
        ce_leave.fit(train, target['outcome'])
        obtained = ce_leave.transform(test)

        self.assertEqual([0.5, 2 / 3.0], list(obtained['color']))

    def test_handle_unknown(self):
        """Should encode unknown values with the global mean."""
        train = pd.Series(['a', 'a', 'a', 'b', 'b', 'b'], name='color')
        target = pd.Series([1.6, 0, 0, 1, 0, 1], name='target')
        test = pd.Series(['b', 'c'], name='color')
        test_target = pd.Series([0, 0])

        ce_leave = encoders.LeaveOneOutEncoder(cols=['color'], handle_unknown='value')
        ce_leave.fit(train, target)
        obtained = ce_leave.transform(test, test_target)

        self.assertEqual([1.0, 0.6], list(obtained['color']))

    def test_leave_one_out_categorical(self):
        """Test that pd.Categorical work the same way as string columns."""
        df = pd.DataFrame(
            {
                'color_str': ['a', 'a', 'a', 'b', 'b', 'b'],
                'color_num_cat': pd.Categorical([1.0, 1.0, 1.0, 2.0, 2.0, 2.0]),
                'color_str_cat': pd.Categorical(['a', 'a', 'a', 'b', 'b', 'b']),
                'outcome': [1, 0, 0, 1, 0, 1],
            }
        )

        X = df.drop('outcome', axis=1)
        y = df['outcome']

        ce_leave = encoders.LeaveOneOutEncoder()
        obtained = ce_leave.fit_transform(X, y)

        for col in obtained.columns:
            self.assertEqual([0.0, 0.5, 0.5, 0.5, 1.0, 0.5], list(obtained[col]))


class TestMixedCategoryTypes(TestCase):
    """Unknown categories need not share the fitted category dtype."""

    def test_unknown_category_policies(self):
        """Apply unknown policies without casting new labels to the fitted dtype."""
        cases = [
            ([1, 1, 2, 2], ['new', 1, None]),
            ([False, False, True, True], ['new', False, None]),
        ]
        for train, query in cases:
            for with_target in (False, True):
                for policy in ('value', 'return_nan', 'error'):
                    with self.subTest(train=train, with_target=with_target, policy=policy):
                        encoder = encoders.LeaveOneOutEncoder(cols=['x'], handle_unknown=policy)
                        encoder.fit(pd.DataFrame({'x': train}), [1, 3, 5, 7])
                        X = pd.DataFrame({'x': query}, index=[8, 2, 8])
                        original = X.copy(deep=True)
                        y = [1, 1, 1] if with_target else None
                        if policy == 'error':
                            with self.assertRaisesRegex(ValueError, 'cannot contain new values'):
                                encoder.transform(X, y)
                        else:
                            actual = encoder.transform(X, y)
                            unknown = 4.0 if policy == 'value' else np.nan
                            expected = pd.DataFrame(
                                {'x': [unknown, 3.0 if with_target else 2.0, 4.0]},
                                index=X.index,
                            )
                            pd.testing.assert_frame_equal(actual, expected)
                        pd.testing.assert_frame_equal(X, original)

    def test_missing_category_keeps_its_fitted_statistics(self):
        """Mixed unknown labels do not change the separate missing-value policy."""
        for missing in ('value', 'return_nan'):
            with self.subTest(missing=missing):
                encoder = encoders.LeaveOneOutEncoder(
                    cols=['x'], handle_unknown='value', handle_missing=missing
                )
                encoder.fit(pd.DataFrame({'x': [1, 1, None, None]}), [1, 3, 5, 7])
                actual = encoder.transform(pd.DataFrame({'x': ['new', 1, np.nan]}))
                expected = pd.DataFrame({'x': [4.0, 2.0, 6.0 if missing == 'value' else np.nan]})
                pd.testing.assert_frame_equal(actual, expected)


class TestCategoryMappingLabels(TestCase):
    """Fitted category labels retain their identity when a missing key is added."""

    def test_non_string_category_labels(self):
        """Keep booleans, large integers and datetimes usable as mapping keys."""
        cases = [
            [False, True],
            [2**60 + 1, 2**60 + 2],
            pd.to_datetime(['2020-01-01', '2020-01-02']),
            pd.date_range('2020-01-01', periods=2, tz='Europe/Paris'),
        ]
        for labels in cases:
            for categorical in (False, True):
                for with_target in (False, True):
                    with self.subTest(labels=labels, categorical=categorical, y=with_target):
                        values = [labels[0], labels[0], labels[1], labels[1]]
                        X = pd.DataFrame({'x': pd.Categorical(values) if categorical else values})
                        before = X.copy(deep=True)
                        encoder = encoders.LeaveOneOutEncoder(cols=['x'])
                        encoder.fit(X, [1, 3, 5, 7])
                        query = X.iloc[[2, 0]].copy()
                        y = [1, 1] if with_target else None
                        actual = encoder.transform(query, y)
                        expected = pd.DataFrame(
                            {'x': [11.0, 3.0] if with_target else [6.0, 2.0]},
                            index=query.index,
                        )
                        pd.testing.assert_frame_equal(actual, expected)
                        pd.testing.assert_frame_equal(X, before)

    def test_fit_transform_preserves_large_integer_categories(self):
        """Adjacent integer labels beyond float precision remain distinct."""
        X = pd.DataFrame({'x': [2**60 + 1, 2**60 + 1, 2**60 + 2, 2**60 + 2]})
        actual = encoders.LeaveOneOutEncoder(cols=['x']).fit_transform(X, [1, 3, 5, 7])
        pd.testing.assert_frame_equal(actual, pd.DataFrame({'x': [3.0, 1.0, 7.0, 5.0]}))

    def test_missing_values_retain_their_statistics(self):
        """Datetime labels keep a separate learned missing group."""
        for labels in (pd.to_datetime(['2020-01-01', '2020-01-02']),):
            for policy in ('value', 'return_nan'):
                with self.subTest(labels=labels, policy=policy):
                    X = pd.DataFrame(
                        {'x': [labels[0], labels[0], labels[1], labels[1], None, None]}
                    )
                    encoder = encoders.LeaveOneOutEncoder(cols=['x'], handle_missing=policy)
                    encoder.fit(X, [1, 3, 5, 7, 9, 11])
                    query = pd.DataFrame({'x': [labels[0], None]})
                    actual = encoder.transform(query)
                    expected = pd.DataFrame({'x': [2.0, 10.0 if policy == 'value' else np.nan]})
                    pd.testing.assert_frame_equal(actual, expected)
class TestCategoricalMissingValues(TestCase):
    """Categorical inputs retain missing values while being encoded."""

    def test_missing_policies(self):
        """Encode missing categories without casting them to a non-nullable dtype."""
        for levels in ([1, 2], [1.0, 2.0], ['a', 'b']):
            for seen_missing in (False, True):
                for with_target in (False, True):
                    for missing in ('value', 'return_nan'):
                        with self.subTest(
                            levels=levels,
                            seen_missing=seen_missing,
                            with_target=with_target,
                            missing=missing,
                        ):
                            values = [levels[0], levels[0], levels[1], levels[1]]
                            target = [1, 3, 5, 7]
                            if seen_missing:
                                values += [None, None]
                                target += [9, 11]
                            X = pd.DataFrame({'x': pd.Categorical(values)})
                            original = X.copy(deep=True)
                            encoder = encoders.LeaveOneOutEncoder(
                                cols=['x'],
                                handle_missing=missing,
                            ).fit(X, target)
                            query = pd.DataFrame(
                                {'x': pd.Categorical([levels[0], None])},
                                index=[7, 2],
                            )
                            before = query.copy(deep=True)
                            y = [1, 1] if with_target else None
                            actual = encoder.transform(query, y)
                            missing_value = (
                                (19.0 if with_target else 10.0)
                                if seen_missing
                                else float(np.mean(target))
                            )
                            if missing == 'return_nan':
                                missing_value = np.nan
                            expected = pd.DataFrame(
                                {'x': [3.0 if with_target else 2.0, missing_value]},
                                index=query.index,
                            )
                            pd.testing.assert_frame_equal(actual, expected)
                            pd.testing.assert_frame_equal(X, original)
                            pd.testing.assert_frame_equal(query, before)

    def test_fit_transform_with_missing_integer_categories(self):
        """Fit and leave each row out when integer categories contain missing values."""
        X = pd.DataFrame({'x': pd.Categorical([1, 1, 2, 2, None, None])})
        encoder = encoders.LeaveOneOutEncoder(cols=['x'])
        actual = encoder.fit_transform(X, [1, 3, 5, 7, 9, 11])
        expected = pd.DataFrame({'x': [3.0, 1.0, 7.0, 5.0, 11.0, 9.0]})
        pd.testing.assert_frame_equal(actual, expected)

    def test_missing_error_policy(self):
        """Reject missing categories with the configured error rather than a cast error."""
        encoder = encoders.LeaveOneOutEncoder(cols=['x'], handle_missing='error')
        encoder.fit(pd.DataFrame({'x': pd.Categorical([1, 1, 2, 2])}), [1, 3, 5, 7])
        with self.assertRaisesRegex(ValueError, 'null'):
            encoder.transform(pd.DataFrame({'x': pd.Categorical([1, None])}))

    def test_unknown_categorical_values(self):
        """Keep unknown and missing policies independent for categorical integers."""
        for policy in ('value', 'return_nan', 'error'):
            with self.subTest(policy=policy):
                encoder = encoders.LeaveOneOutEncoder(cols=['x'], handle_unknown=policy)
                encoder.fit(pd.DataFrame({'x': pd.Categorical([1, 1, 2, 2])}), [1, 3, 5, 7])
                query = pd.DataFrame({'x': pd.Categorical([3, 1, None])})
                if policy == 'error':
                    with self.assertRaisesRegex(ValueError, 'cannot contain new values'):
                        encoder.transform(query)
                else:
                    unknown = 4.0 if policy == 'value' else np.nan
                    expected = pd.DataFrame({'x': [unknown, 2.0, 4.0]})
                    pd.testing.assert_frame_equal(encoder.transform(query), expected)
