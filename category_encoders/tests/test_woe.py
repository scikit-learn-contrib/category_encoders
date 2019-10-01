import pandas as pd
from unittest2 import TestCase  # or `from unittest import ...` if on Python 3.4+
import category_encoders.tests.helpers as th
import numpy as np

import category_encoders as encoders

np_X = th.create_array(n_rows=100)
np_X_t = th.create_array(n_rows=50, extras=True)
np_y = np.random.randn(np_X.shape[0]) > 0.5
np_y_t = np.random.randn(np_X_t.shape[0]) > 0.5
X = th.create_dataset(n_rows=100)
X_t = th.create_dataset(n_rows=50, extras=True)
y = pd.DataFrame(np_y)
y_t = pd.DataFrame(np_y_t)


class TestWeightOfEvidenceEncoder(TestCase):
    def test_woe(self):
        cols = ['unique_str', 'underscore', 'extra', 'none', 'invariant', 321, 'categorical', 'na_categorical', 'categorical_int']

        # balanced label with balanced features
        X_balanced = pd.DataFrame(data=['1', '1', '1', '2', '2', '2'], columns=['col1'])
        y_balanced = [True, False, True, False, True, False]
        enc = encoders.WOEEncoder()
        enc.fit(X_balanced, y_balanced)
        X1 = enc.transform(X_balanced)
        self.assertTrue(all(X1.sum() < 0.001),
                        "When the class label is balanced, WoE should sum to 0 in each transformed column")

        enc = encoders.WOEEncoder(cols=cols)
        enc.fit(X, np_y)
        X1 = enc.transform(X_t)
        th.verify_numeric(X1[cols])
        self.assertTrue(np.isfinite(X1[cols].values).all(),
                        'There must not be any NaN, inf or -inf in the transformed columns')
        self.assertEqual(len(list(X_t)), len(list(X1)), 'The count of attributes must not change')
        self.assertEqual(len(X_t), len(X1), 'The count of rows must not change')
        X2 = enc.transform(X_t, np_y_t)
        th.verify_numeric(X2)
        self.assertTrue(np.isfinite(X2[cols].values).all(),
                        'There must not be any NaN, inf or -inf in the transformed columns')
        self.assertEqual(len(list(X_t)), len(list(X2)), 'The count of attributes must not change')
        self.assertEqual(len(X_t), len(X2), 'The count of rows must not change')
        X3 = enc.transform(X, np_y)
        th.verify_numeric(X3)
        self.assertTrue(np.isfinite(X3[cols].values).all(),
                        'There must not be any NaN, inf or -inf in the transformed columns')
        self.assertEqual(len(list(X)), len(list(X3)), 'The count of attributes must not change')
        self.assertEqual(len(X), len(X3), 'The count of rows must not change')
        self.assertTrue(X3['unique_str'].var() < 0.001, 'The unique string column must not be predictive of the label')
        X4 = enc.fit_transform(X, np_y)
        th.verify_numeric(X4)
        self.assertTrue(np.isfinite(X4[cols].values).all(),
                        'There must not be any NaN, inf or -inf in the transformed columns')
        self.assertEqual(len(list(X)), len(list(X4)), 'The count of attributes must not change')
        self.assertEqual(len(X), len(X4), 'The count of rows must not change')
        self.assertTrue(X4['unique_str'].var() < 0.001, 'The unique string column must not be predictive of the label')

        enc = encoders.WOEEncoder()
        enc.fit(X, np_y)
        X1 = enc.transform(X_t)
        self.assertEqual(len(list(X_t)), len(list(X1)), 'The count of attributes must not change')
        self.assertEqual(len(X_t), len(X1), 'The count of rows must not change')
        th.verify_numeric(X1)
        X2 = enc.transform(X_t, np_y_t)
        th.verify_numeric(X2)
        self.assertEqual(len(list(X_t)), len(list(X2)), 'The count of attributes must not change')
        self.assertEqual(len(X_t), len(X2), 'The count of rows must not change')

        # seed
        enc = encoders.WOEEncoder(cols=cols, random_state=2001, randomized=True)
        enc.fit(X, np_y)
        X1 = enc.transform(X_t, np_y_t)
        X2 = enc.transform(X_t, np_y_t)
        self.assertTrue(X1.equals(X2), "When the seed is given, the results must be identical")
        th.verify_numeric(X1)
        th.verify_numeric(X2)

        # invariant target
        y_invariant = [True, True, True, True, True, True]
        enc = encoders.WOEEncoder()
        with self.assertRaises(ValueError):
            enc.fit(X_balanced, y_invariant)

        # branch coverage unit tests - no cols
        enc = encoders.WOEEncoder(cols=[])
        enc.fit(X, np_y)
        self.assertTrue(enc.transform(X_t).equals(X_t))

        # missing values in the target
        y_missing = [True, True, None, True, True, True]
        enc = encoders.WOEEncoder()
        with self.assertRaises(ValueError):
            enc.fit(X_balanced, y_missing)

        # impute missing
        enc = encoders.WOEEncoder(handle_missing='return_nan')
        enc.fit(X, np_y)
        X1 = enc.transform(X_t)
        th.verify_numeric(X1)
        self.assertTrue(X1.isnull().values.any())
        self.assertEqual(len(list(X_t)), len(list(X1)), 'The count of attributes must not change')
        self.assertEqual(len(X_t), len(X1), 'The count of rows must not change')

        X2 = enc.transform(X_t, np_y_t)
        th.verify_numeric(X2)
        self.assertTrue(X1.isnull().values.any())
        self.assertEqual(len(list(X_t)), len(list(X2)), 'The count of attributes must not change')
        self.assertEqual(len(X_t), len(X2), 'The count of rows must not change')

    def test_HaveArrays_ExpectCalculatedProperly(self):
        X = ['a', 'a', 'b', 'b']
        y = [1, 0, 0, 0]
        enc = encoders.WOEEncoder()

        result = enc.fit_transform(X, y)

        expected = pd.Series([0.5108256237659906, .5108256237659906, -0.587786664902119, -0.587786664902119], name=0)
        pd.testing.assert_series_equal(expected, result[0])

    def test_HandleMissingValue_HaveMissingInTrain_ExpectEncoded(self):
        X = ['a', 'a', np.nan, np.nan]
        y = [1, 0, 0, 0]
        enc = encoders.WOEEncoder(handle_missing='value')

        result = enc.fit_transform(X, y)

        expected = pd.Series([0.5108256237659906, .5108256237659906, -0.587786664902119, -0.587786664902119], name=0)
        pd.testing.assert_series_equal(expected, result[0])

    def test_HandleMissingValue_HaveMissingInTest_ExpectEncodedWithZero(self):
        X = ['a', 'a', 'b', 'b']
        y = [1, 0, 0, 0]
        test = ['a', np.nan]
        enc = encoders.WOEEncoder(handle_missing='value')

        enc.fit(X, y)
        result = enc.transform(test)

        expected = pd.Series([0.5108256237659906, 0], name=0)
        pd.testing.assert_series_equal(expected, result[0])

    def test_HandleUnknownValue_HaveUnknown_ExpectEncodedWithZero(self):
        X = ['a', 'a', 'b', 'b']
        y = [1, 0, 0, 0]
        test = ['a', 'c']
        enc = encoders.WOEEncoder(handle_unknown='value')

        enc.fit(X, y)
        result = enc.transform(test)

        expected = pd.Series([0.5108256237659906, 0], name=0)
        pd.testing.assert_series_equal(expected, result[0])
