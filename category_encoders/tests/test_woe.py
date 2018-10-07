import pandas as pd
from unittest2 import TestCase  # or `from unittest import ...` if on Python 3.4+
import category_encoders.tests.test_utils as tu
import numpy as np

import category_encoders as encoders

np_X = tu.create_array(n_rows=100)
np_X_t = tu.create_array(n_rows=50, extras=True)
np_y = np.random.randn(np_X.shape[0]) > 0.5
np_y_t = np.random.randn(np_X_t.shape[0]) > 0.5
X = tu.create_dataset(n_rows=100)
X_t = tu.create_dataset(n_rows=50, extras=True)
y = pd.DataFrame(np_y)
y_t = pd.DataFrame(np_y_t)


class TestWeightOfEvidenceEncoder(TestCase):
    def test_woe(self):
        cols = ['unique_str', 'underscore', 'extra', 'none', 'invariant', 321]

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
        tu.verify_numeric(X1[cols])
        self.assertTrue(np.isfinite(X1[cols].values).all(),
                        'There must not be any NaN, inf or -inf in the transformed columns')
        self.assertEqual(len(list(X_t)), len(list(X1)), 'The count of attributes must not change')
        self.assertEqual(len(X_t), len(X1), 'The count of rows must not change')
        X2 = enc.transform(X_t, np_y_t)
        tu.verify_numeric(X2)
        self.assertTrue(np.isfinite(X2[cols].values).all(),
                        'There must not be any NaN, inf or -inf in the transformed columns')
        self.assertEqual(len(list(X_t)), len(list(X2)), 'The count of attributes must not change')
        self.assertEqual(len(X_t), len(X2), 'The count of rows must not change')
        X3 = enc.transform(X, np_y)
        tu.verify_numeric(X3)
        self.assertTrue(np.isfinite(X3[cols].values).all(),
                        'There must not be any NaN, inf or -inf in the transformed columns')
        self.assertEqual(len(list(X)), len(list(X3)), 'The count of attributes must not change')
        self.assertEqual(len(X), len(X3), 'The count of rows must not change')
        self.assertTrue(X3['unique_str'].var() < 0.001, 'The unique string column must not be predictive of the label')
        X4 = enc.fit_transform(X, np_y)
        tu.verify_numeric(X4)
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
        tu.verify_numeric(X1)
        X2 = enc.transform(X_t, np_y_t)
        tu.verify_numeric(X2)
        self.assertEqual(len(list(X_t)), len(list(X2)), 'The count of attributes must not change')
        self.assertEqual(len(X_t), len(X2), 'The count of rows must not change')

        # seed
        enc = encoders.WOEEncoder(cols=cols, random_state=2001, randomized=True)
        enc.fit(X, np_y)
        X1 = enc.transform(X_t, np_y_t)
        X2 = enc.transform(X_t, np_y_t)
        self.assertTrue(X1.equals(X2), "When the seed is given, the results must be identical")
        tu.verify_numeric(X1)
        tu.verify_numeric(X2)

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
        enc = encoders.WOEEncoder(impute_missing=False)
        enc.fit(X, np_y)
        X1 = enc.transform(X_t)
        tu.verify_numeric(X1)
        self.assertTrue(X1.isnull().values.any())
        self.assertEqual(len(list(X_t)), len(list(X1)), 'The count of attributes must not change')
        self.assertEqual(len(X_t), len(X1), 'The count of rows must not change')

        X2 = enc.transform(X_t, np_y_t)
        tu.verify_numeric(X2)
        self.assertTrue(X1.isnull().values.any())
        self.assertEqual(len(list(X_t)), len(list(X2)), 'The count of attributes must not change')
        self.assertEqual(len(X_t), len(X2), 'The count of rows must not change')
