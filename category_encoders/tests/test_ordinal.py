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


class TestOrdinalEncoder(TestCase):

    def test_ordinal(self):

        enc = encoders.OrdinalEncoder(verbose=1, return_df=True, impute_missing=True)
        enc.fit(X)
        out = enc.transform(X_t)
        self.assertEqual(len(set(out['extra'].values)), 4)
        self.assertIn(0, set(out['extra'].values))
        self.assertFalse(enc.mapping is None)
        self.assertTrue(len(enc.mapping) > 0)

        enc = encoders.OrdinalEncoder(verbose=1, mapping=enc.mapping, return_df=True, impute_missing=True)
        enc.fit(X)
        out = enc.transform(X_t)
        self.assertEqual(len(set(out['extra'].values)), 4)
        self.assertIn(0, set(out['extra'].values))
        self.assertTrue(len(enc.mapping) > 0)

        enc = encoders.OrdinalEncoder(verbose=1, return_df=True, impute_missing=True, handle_unknown='ignore')
        enc.fit(X)
        out = enc.transform(X_t)
        out_cats = [x for x in set(out['extra'].values) if np.isfinite(x)]
        self.assertEqual(len(out_cats), 3)
        self.assertFalse(enc.mapping is None)

    def test_ordinal_dist(self):
        data = np.array([
            ['apple', None],
            ['peach', 'lemon']
        ])
        encoder = encoders.OrdinalEncoder(impute_missing=True)
        encoder.fit(data)
        a = encoder.transform(data)
        self.assertEqual(a.values[0, 1], 0)
        self.assertEqual(a.values[1, 1], 1)
        pd.factorize
        encoder = encoders.OrdinalEncoder(impute_missing=False)
        encoder.fit(data)
        a = encoder.transform(data)
        self.assertTrue(np.isnan(a.values[0, 1]))
        self.assertEqual(a.values[1, 1], 1)

    def test_pandas_categorical(self):
        X = pd.DataFrame({
            'Str': ['a', 'c', 'c', 'd'],
            'Categorical': pd.Categorical(list('bbea'), categories=['e', 'a', 'b'], ordered=True)
        })

        enc = encoders.OrdinalEncoder()
        out = enc.fit_transform(X)

        tu.verify_numeric(out)
        self.assertEqual(3, out['Categorical'][0])
        self.assertEqual(3, out['Categorical'][1])
        self.assertEqual(1, out['Categorical'][2])
        self.assertEqual(2, out['Categorical'][3])
