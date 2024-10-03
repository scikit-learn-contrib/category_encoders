"""Tests for the wrapper module."""
from unittest import TestCase

import category_encoders as encoders
import numpy as np
import pandas as pd
from category_encoders.wrapper import NestedCVWrapper, PolynomialWrapper
from sklearn.model_selection import GroupKFold

import tests.helpers as th


class TestMultiClassWrapper(TestCase):
    """Tests for the PolynomialWrapper class."""

    def test_invariance_to_data_types(self):
        """Test that the wrapper is invariant to data types."""
        x = np.array(
            [
                ['a', 'b', 'c'],
                ['a', 'b', 'c'],
                ['b', 'b', 'c'],
                ['b', 'b', 'b'],
                ['b', 'b', 'b'],
                ['a', 'b', 'a'],
            ]
        )
        y = [1, 2, 3, 3, 3, 3]
        wrapper = PolynomialWrapper(encoders.TargetEncoder())
        result = wrapper.fit_transform(x, y)
        th.verify_numeric(result)

        x = pd.DataFrame(
            [
                ['a', 'b', 'c'],
                ['a', 'b', 'c'],
                ['b', 'b', 'c'],
                ['b', 'b', 'b'],
                ['b', 'b', 'b'],
                ['a', 'b', 'a'],
            ],
            columns=['f1', 'f2', 'f3'],
        )
        y = ['bee', 'cat', 'dog', 'dog', 'dog', 'dog']
        wrapper = PolynomialWrapper(encoders.TargetEncoder())
        result2 = wrapper.fit_transform(x, y)

        self.assertTrue(
            (result.to_numpy() == result2.to_numpy()).all(),
            'The content should be the same regardless whether we pass Numpy or Pandas data type.',
        )

    def test_transform_only_selected(self):
        """Test that the wrapper only transforms the selected columns."""
        x = pd.DataFrame(
            [
                ['a', 'b', 'c'],
                ['a', 'a', 'c'],
                ['b', 'a', 'c'],
                ['b', 'c', 'b'],
                ['b', 'b', 'b'],
                ['a', 'b', 'a'],
            ],
            columns=['f1', 'f2', 'f3'],
        )
        y = ['bee', 'cat', 'dog', 'dog', 'dog', 'dog']
        wrapper = PolynomialWrapper(encoders.LeaveOneOutEncoder(cols=['f2']))

        # combination fit() + transform()
        wrapper.fit(x, y)
        result = wrapper.transform(x, y)
        self.assertEqual(
            len(result.columns),
            4,
            'We expect 2 untouched features + f2 target encoded into 2 features',
        )

        # directly fit_transform()
        wrapper = PolynomialWrapper(encoders.LeaveOneOutEncoder(cols=['f2']))
        result2 = wrapper.fit_transform(x, y)
        self.assertEqual(
            len(result2.columns),
            4,
            'We expect 2 untouched features + f2 target encoded into 2 features',
        )

        pd.testing.assert_frame_equal(result, result2)

    def test_refit_stateless(self):
        """Test that when the encoder is fitted multiple times no old state is carried."""
        x = pd.DataFrame(
            [
                ['a', 'b', 'c'],
                ['a', 'b', 'c'],
                ['b', 'b', 'c'],
                ['b', 'b', 'b'],
                ['b', 'b', 'b'],
                ['a', 'b', 'a'],
            ],
            columns=['f1', 'f2', 'f3'],
        )
        y1 = ['bee', 'cat', 'dog', 'dog', 'dog', 'dog']
        y2 = ['bee', 'cat', 'duck', 'duck', 'duck', 'duck']
        wrapper = PolynomialWrapper(encoders.TargetEncoder())
        _ = wrapper.fit_transform(x, y1)
        expected_categories_1 = {
            'cat',
            'dog',
        }  # 'bee' is dropped since first label is always dropped
        expected_categories_2 = {'cat', 'duck'}
        self.assertEqual(
            set(wrapper.label_encoder.ordinal_encoder.category_mapping[0]['mapping'].index),
            {'bee', 'cat', 'dog'},
        )
        self.assertEqual(set(wrapper.feature_encoders.keys()), expected_categories_1)
        _ = wrapper.fit_transform(x, y2)
        self.assertEqual(
            set(wrapper.label_encoder.ordinal_encoder.category_mapping[0]['mapping'].index),
            {'bee', 'cat', 'duck'},
        )
        self.assertEqual(set(wrapper.feature_encoders.keys()), expected_categories_2)


class TestNestedCVWrapper(TestCase):
    """Tests for the NestedCVWrapper class."""

    def test_train_not_equal_to_valid(self):
        """Test that the train and valid results are not equal."""
        x = np.array(
            [
                ['a', 'b', 'c'],
                ['a', 'b', 'c'],
                ['a', 'b', 'c'],
                ['a', 'b', 'c'],
                ['b', 'b', 'c'],
                ['b', 'b', 'c'],
                ['b', 'b', 'b'],
                ['b', 'b', 'b'],
                ['b', 'b', 'b'],
                ['b', 'b', 'b'],
                ['a', 'b', 'a'],
                ['a', 'b', 'a'],
            ]
        )
        y = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
        wrapper = NestedCVWrapper(encoders.TargetEncoder(), cv=3)
        result_train, result_valid = wrapper.fit_transform(x, y, X_test=x)

        # We would expect result_train != result_valid since result_train has been generated using
        # nested # folds and result_valid is generated by fitting the encoder on all the x & y data
        self.assertFalse(np.allclose(result_train, result_valid))

    def test_custom_cv(self):
        """Test custom cross validation."""
        x = np.array(
            [
                ['a', 'b', 'c'],
                ['a', 'b', 'c'],
                ['a', 'b', 'c'],
                ['a', 'b', 'c'],
                ['b', 'b', 'c'],
                ['b', 'b', 'c'],
                ['b', 'b', 'b'],
                ['b', 'b', 'b'],
                ['b', 'b', 'b'],
                ['b', 'b', 'b'],
                ['a', 'b', 'a'],
                ['a', 'b', 'a'],
            ]
        )
        groups = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]
        y = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
        gkfold = GroupKFold(n_splits=3)
        wrapper = NestedCVWrapper(encoders.TargetEncoder(), cv=gkfold)
        result_train, result_valid = wrapper.fit_transform(x, y, X_test=x, groups=groups)

        # We would expect result_train != result_valid since result_train has been generated using
        # nested # folds and result_valid is generated by fitting the encoder on all the x & y data
        self.assertFalse(np.allclose(result_train, result_valid))
