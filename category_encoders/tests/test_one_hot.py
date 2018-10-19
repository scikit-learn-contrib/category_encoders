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


class TestOneHotEncoderTestCase(TestCase):

    def test_one_hot(self):
        enc = encoders.OneHotEncoder(verbose=1, return_df=False)
        enc.fit(X)
        self.assertEqual(enc.transform(X_t).shape[1],
                         enc.transform(X_t[X_t['extra'] != 'A']).shape[1],
                         'We have to get the same count of columns')

        enc = encoders.OneHotEncoder(
            verbose=1, return_df=True, impute_missing=True)
        enc.fit(X)
        out = enc.transform(X_t)
        self.assertIn('extra_-1', out.columns.values)

        enc = encoders.OneHotEncoder(
            verbose=1, return_df=True, impute_missing=True, handle_unknown='ignore')
        enc.fit(X)
        out = enc.transform(X_t)
        self.assertEqual(
            len([x for x in out.columns.values if str(x).startswith('extra_')]), 3)

        enc = encoders.OneHotEncoder(
            verbose=1, return_df=True, impute_missing=True, handle_unknown='error')
        enc.fit(X)
        with self.assertRaises(ValueError):
            out = enc.transform(X_t)

        enc = encoders.OneHotEncoder(
            verbose=1, return_df=True, handle_unknown='ignore', use_cat_names=True)
        enc.fit(X)
        out = enc.transform(X_t)
        self.assertIn('extra_A', out.columns.values)

        enc = encoders.OneHotEncoder(
            verbose=1, return_df=True, use_cat_names=True)
        enc.fit(X)
        out = enc.transform(X_t)
        self.assertIn('extra_-1', out.columns.values)

        # test get_feature_names
        enc = encoders.OneHotEncoder(
            verbose=1, return_df=True, use_cat_names=True)
        enc.fit(X)
        out = enc.transform(X_t)
        self.assertEqual(set(enc.get_feature_names()), set(out.columns))

        # test inverse_transform
        X_i = tu.create_dataset(n_rows=100, has_none=False)
        X_i_t = tu.create_dataset(n_rows=50, has_none=False)
        X_i_t_extra = tu.create_dataset(n_rows=50, extras=True, has_none=False)
        cols = ['underscore', 'none', 'extra', 321]

        enc = encoders.OneHotEncoder(verbose=1, use_cat_names=True, cols=cols)
        enc.fit(X_i)
        obtained = enc.inverse_transform(enc.transform(X_i_t))
        # numeric columns are incorrectly typed as object...
        obtained[321] = obtained[321].astype('int64')
        tu.verify_inverse_transform(X_i_t, obtained)
