import doctest
import math
import os
import random
import unittest
import numpy as np
import pandas as pd

import category_encoders as encoders

__author__ = 'willmcginnis'



class TestEncoders(unittest.TestCase):

    def verify_numeric(self, X_test):
        for dt in X_test.dtypes:
            numeric = False
            if np.issubdtype(dt, np.dtype(int)) or np.issubdtype(dt, np.dtype(float)):
                numeric = True
            self.assertTrue(numeric)

    def verify_inverse_transform(self, x, x_inv):
        """
        Verify x is equal to x_inv. The test returns true for NaN.equals(NaN) as it should.

        """
        self.assertTrue(x.equals(x_inv))

    def create_dataset(self, n_rows=1000, extras=False, has_none=True):
        """
        Creates a dataset with some categorical variables
        """

        ds = [[
            random.random(),                                                                        # Floats
            random.choice([float('nan'), float('inf'), float('-inf'), -0, 0, 1, -1, math.pi]),      # Floats with edge scenarios
            row,                                                                                    # Unique integers
            random.choice([12, 43, -32]),                                                           # Numbers everywhere
            str(row),                                                                               # Unique strings
            random.choice(['A', 'B_b', 'C_c_c']),                                                   # Strings with underscores to test reverse_dummies()
            random.choice(['A', 'B', 'C', 'D']) if extras else random.choice(['A', 'B', 'C']),      # With a new string value
            random.choice(['A', 'B', 'C', None]) if has_none else random.choice(['A', 'B', 'C']),   # None
            random.choice(['A'])                                                                    # Invariant

        ] for row in range(n_rows)]

        df = pd.DataFrame(ds, columns=['float', 'float_edge', 'unique_int', 321, 'unique_str', 'underscore', 'extra', 'none', 'invariant'])
        return df

    def create_array(self, n_rows=1000, extras=False, has_none=True):
        """
        Creates a numpy dataset with some categorical variables
        :return:
        """

        ds = [[
                  random.random(),
                  random.random(),
                  random.choice(['A', 'B', 'C']),
                  random.choice(['A', 'B', 'C', 'D']) if extras else random.choice(['A', 'B', 'C']),
                  random.choice(['A', 'B', 'C', None, np.nan]) if has_none else random.choice(['A', 'B', 'C']),
                  random.choice(['A'])
              ] for _ in range(n_rows)]

        return np.array(ds)


    def test_hashing_np(self):
        """
        Creates a dataset and encodes with with the hashing trick

        :return:
        """

        X = self.create_array()
        X_t = self.create_array(n_rows=100)

        enc = encoders.HashingEncoder(verbose=1, n_components=128)
        enc.fit(X)
        self.verify_numeric(enc.transform(X_t))

    def test_hashing(self):
        """
        Creates a dataset and encodes with with the hashing trick

        :return:
        """

        cols = ['unique_str', 'underscore', 'extra', 'none', 'invariant', 321]
        X = self.create_dataset()
        X_t = self.create_dataset(n_rows=100)

        enc = encoders.HashingEncoder(verbose=1, n_components=128, cols=cols)
        enc.fit(X)
        self.verify_numeric(enc.transform(X_t))

        enc = encoders.HashingEncoder(verbose=1, n_components=32)
        enc.fit(X)
        self.verify_numeric(enc.transform(X_t))

        enc = encoders.HashingEncoder(verbose=1, n_components=32, drop_invariant=True)
        enc.fit(X)
        self.verify_numeric(enc.transform(X_t))

        enc = encoders.HashingEncoder(verbose=1, n_components=32, return_df=False)
        enc.fit(X)
        self.assertTrue(isinstance(enc.transform(X_t), np.ndarray))

    def test_ordinal_np(self):
        """

        :return:
        """

        X = self.create_array()
        X_t = self.create_array(n_rows=100)

        enc = encoders.OrdinalEncoder(verbose=1)
        enc.fit(X)
        self.verify_numeric(enc.transform(X_t))

    def test_ordinal(self):
        """

        :return:
        """

        cols = ['unique_str', 'underscore', 'extra', 'none', 'invariant', 321]
        X = self.create_dataset()
        X_t = self.create_dataset(n_rows=100)
        X_t_extra = self.create_dataset(n_rows=100, extras=True)

        enc = encoders.OrdinalEncoder(verbose=1, cols=cols)
        enc.fit(X)
        self.verify_numeric(enc.transform(X_t))

        enc = encoders.OrdinalEncoder(verbose=1)
        enc.fit(X)
        self.verify_numeric(enc.transform(X_t))

        enc = encoders.OrdinalEncoder(verbose=1, drop_invariant=True)
        enc.fit(X)
        self.verify_numeric(enc.transform(X_t))

        enc = encoders.OrdinalEncoder(verbose=1, return_df=False)
        enc.fit(X)
        self.assertTrue(isinstance(enc.transform(X_t), np.ndarray))

        enc = encoders.OrdinalEncoder(verbose=1, return_df=True, impute_missing=True)
        enc.fit(X)
        out = enc.transform(X_t_extra)
        self.assertEqual(len(set(out['extra'].values)), 4)
        self.assertIn(0, set(out['extra'].values))
        self.assertFalse(enc.mapping is None)
        self.assertTrue(len(enc.mapping) > 0)

        enc = encoders.OrdinalEncoder(verbose=1, mapping=enc.mapping, return_df=True, impute_missing=True)
        enc.fit(X)
        out = enc.transform(X_t_extra)
        self.assertEqual(len(set(out['extra'].values)), 4)
        self.assertIn(0, set(out['extra'].values))
        self.assertTrue(len(enc.mapping) > 0)

        enc = encoders.OrdinalEncoder(verbose=1, return_df=True, impute_missing=True, handle_unknown='ignore')
        enc.fit(X)
        out = enc.transform(X_t_extra)
        out_cats = [x for x in set(out['extra'].values) if np.isfinite(x)]
        self.assertEqual(len(out_cats), 3)
        self.assertFalse(enc.mapping is None)

        enc = encoders.OrdinalEncoder(verbose=1, return_df=True, handle_unknown='error')
        enc.fit(X)
        with self.assertRaises(ValueError):
            out = enc.transform(X_t_extra)

        # test inverse_transform
        X = self.create_dataset(has_none=False)
        X_t = self.create_dataset(n_rows=100, has_none=False)
        X_t_extra = self.create_dataset(n_rows=100, extras=True, has_none=False)

        enc = encoders.OrdinalEncoder(verbose=1)
        enc.fit(X)
        self.verify_numeric(enc.transform(X_t))
        self.verify_inverse_transform(X_t, enc.inverse_transform(enc.transform(X_t)))
        with self.assertRaises(ValueError):
            out = enc.inverse_transform(enc.transform(X_t_extra))

    def test_backward_difference_np(self):
        """

        :return:
        """

        X = self.create_array()
        X_t = self.create_array(n_rows=100)

        enc = encoders.BackwardDifferenceEncoder(verbose=1)
        enc.fit(X)
        self.verify_numeric(enc.transform(X_t))

    def test_backward_difference(self):
        """

        :return:
        """

        cols = ['unique_str', 'underscore', 'extra', 'none', 'invariant', 321]
        X = self.create_dataset()
        X_t = self.create_dataset(n_rows=100)

        enc = encoders.BackwardDifferenceEncoder(verbose=1, cols=cols)
        enc.fit(X)
        self.verify_numeric(enc.transform(X_t))

        enc = encoders.BackwardDifferenceEncoder(verbose=1)
        enc.fit(X)
        self.verify_numeric(enc.transform(X_t))

        enc = encoders.BackwardDifferenceEncoder(verbose=1, drop_invariant=True)
        enc.fit(X)
        self.verify_numeric(enc.transform(X_t))

        enc = encoders.BackwardDifferenceEncoder(verbose=1, return_df=False)
        enc.fit(X)
        self.assertTrue(isinstance(enc.transform(X_t), np.ndarray))

    def test_binary_np(self):
        """

        :return:
        """

        X = self.create_array()
        X_t = self.create_array(n_rows=100)

        enc = encoders.BinaryEncoder(verbose=1)
        enc.fit(X)
        self.verify_numeric(enc.transform(X_t))

    def test_binary(self):
        """

        :return:
        """

        cols = ['unique_str', 'underscore', 'extra', 'none', 'invariant', 321]
        X = self.create_dataset()
        X_t = self.create_dataset(n_rows=100)

        enc = encoders.BinaryEncoder(verbose=1, cols=cols)
        enc.fit(X)
        self.verify_numeric(enc.transform(X_t))

        enc = encoders.BinaryEncoder(verbose=1)
        enc.fit(X)
        self.verify_numeric(enc.transform(X_t))

        enc = encoders.BinaryEncoder(verbose=1, drop_invariant=True)
        enc.fit(X)
        self.verify_numeric(enc.transform(X_t))

        enc = encoders.BinaryEncoder(verbose=1, return_df=False)
        enc.fit(X)
        self.assertTrue(isinstance(enc.transform(X_t), np.ndarray))

        # test inverse_transform
        X = self.create_dataset(has_none=False)
        X_t = self.create_dataset(n_rows=100, has_none=False)
        X_t_extra = self.create_dataset(n_rows=100, extras=True, has_none=False)

        enc = encoders.BinaryEncoder(verbose=1)
        enc.fit(X)
        self.verify_numeric(enc.transform(X_t))
        self.verify_inverse_transform(X_t, enc.inverse_transform(enc.transform(X_t)))
        with self.assertRaises(ValueError):
            out = enc.inverse_transform(enc.transform(X_t_extra))

    def test_basen_np(self):
        """

        :return:
        """

        X = self.create_array()
        X_t = self.create_array(n_rows=100)

        enc = encoders.BaseNEncoder(verbose=1)
        enc.fit(X)
        self.verify_numeric(enc.transform(X_t))

    def test_basen(self):

        cols = ['unique_str', 'underscore', 'extra', 'none', 'invariant', 321]
        X = self.create_dataset()
        X_t = self.create_dataset(n_rows=100)
        X_t_extra = self.create_dataset(n_rows=100, extras=True)

        enc = encoders.BaseNEncoder(verbose=1, cols=cols)
        enc.fit(X)
        self.verify_numeric(enc.transform(X_t))

        enc = encoders.BaseNEncoder(verbose=1)
        enc.fit(X)
        self.verify_numeric(enc.transform(X_t))

        enc = encoders.BaseNEncoder(verbose=1, drop_invariant=True)
        enc.fit(X)
        self.verify_numeric(enc.transform(X_t))

        enc = encoders.BaseNEncoder(verbose=1, return_df=False)
        enc.fit(X)
        self.assertTrue(isinstance(enc.transform(X_t), np.ndarray))

        # test inverse_transform
        X = self.create_dataset(has_none=False)
        X_t = self.create_dataset(n_rows=100, has_none=False)
        X_t_extra = self.create_dataset(n_rows=100, extras=True, has_none=False)

        enc = encoders.BaseNEncoder(verbose=1, impute_missing=False, base=1)
        enc.fit(X)
        self.verify_numeric(enc.transform(X_t))
        self.verify_inverse_transform(X_t, enc.inverse_transform(enc.transform(X_t)))

        # enc = encoders.BaseNEncoder(verbose=1, cols=['D'], drop_invariant=True)
        # enc.fit(X)
        # self.verify_inverse_transform(X_t, enc.inverse_transform(enc.transform(X_t)))

    def test_helmert_np(self):
        """

        :return:
        """

        X = self.create_array()
        X_t = self.create_array(n_rows=100)

        enc = encoders.HelmertEncoder(verbose=1)
        enc.fit(X)
        self.verify_numeric(enc.transform(X_t))

    def test_helmert(self):
        """

        :return:
        """

        cols = ['unique_str', 'underscore', 'extra', 'none', 'invariant', 321]
        X = self.create_dataset()
        X_t = self.create_dataset(n_rows=100)

        enc = encoders.HelmertEncoder(verbose=1, cols=cols)
        enc.fit(X)
        self.verify_numeric(enc.transform(X_t))

        enc = encoders.HelmertEncoder(verbose=1)
        enc.fit(X)
        self.verify_numeric(enc.transform(X_t))

        enc = encoders.HelmertEncoder(verbose=1, drop_invariant=True)
        enc.fit(X)
        self.verify_numeric(enc.transform(X_t))

        enc = encoders.HelmertEncoder(verbose=1, return_df=False)
        enc.fit(X)
        self.assertTrue(isinstance(enc.transform(X_t), np.ndarray))

    def test_polynomial_np(self):
        """

        :return:
        """

        X = self.create_array()
        X_t = self.create_array(n_rows=100)

        enc = encoders.PolynomialEncoder(verbose=1)
        enc.fit(X)
        self.verify_numeric(enc.transform(X_t))

    def test_polynomial(self):
        """

        :return:
        """

        cols = ['unique_str', 'underscore', 'extra', 'none', 'invariant', 321]
        X = self.create_dataset()
        X_t = self.create_dataset(n_rows=100)

        enc = encoders.PolynomialEncoder(verbose=1, cols=cols)
        enc.fit(X)
        self.verify_numeric(enc.transform(X_t))

        enc = encoders.PolynomialEncoder(verbose=1)
        enc.fit(X)
        self.verify_numeric(enc.transform(X_t))

        enc = encoders.PolynomialEncoder(verbose=1, drop_invariant=True)
        enc.fit(X)
        self.verify_numeric(enc.transform(X_t))

        enc = encoders.PolynomialEncoder(verbose=1, return_df=False)
        enc.fit(X)
        self.assertTrue(isinstance(enc.transform(X_t), np.ndarray))

    def test_sum_np(self):
        """

        :return:
        """

        X = self.create_array()
        X_t = self.create_array(n_rows=100)

        enc = encoders.SumEncoder(verbose=1)
        enc.fit(X)
        self.verify_numeric(enc.transform(X_t))

    def test_sum(self):
        """

        :return:
        """

        cols = ['unique_str', 'underscore', 'extra', 'none', 'invariant', 321]
        X = self.create_dataset()
        X_t = self.create_dataset(n_rows=100)

        enc = encoders.SumEncoder(verbose=1, cols=cols)
        enc.fit(X)
        self.verify_numeric(enc.transform(X_t))

        enc = encoders.SumEncoder(verbose=1)
        enc.fit(X)
        self.verify_numeric(enc.transform(X_t))

        enc = encoders.SumEncoder(verbose=1, drop_invariant=True)
        enc.fit(X)
        self.verify_numeric(enc.transform(X_t))

        enc = encoders.SumEncoder(verbose=1, return_df=False)
        enc.fit(X)
        self.assertTrue(isinstance(enc.transform(X_t), np.ndarray))

    def test_one_hot_np(self):
        """

        :return:
        """

        X = self.create_array()
        X_t = self.create_array(n_rows=100)

        enc = encoders.OneHotEncoder(verbose=1)
        enc.fit(X)
        self.verify_numeric(enc.transform(X_t))

    def test_one_hot(self):

        cols = ['unique_str', 'underscore', 'extra', 'none', 'invariant', 321]
        X = self.create_dataset()
        X_t = self.create_dataset(n_rows=100)
        X_t_extra = self.create_dataset(n_rows=100, extras=True)

        enc = encoders.OneHotEncoder(verbose=1, cols=cols)
        enc.fit(X)
        self.verify_numeric(enc.transform(X_t))

        enc = encoders.OneHotEncoder(verbose=1)
        enc.fit(X)
        self.verify_numeric(enc.transform(X_t))

        enc = encoders.OneHotEncoder(verbose=1, drop_invariant=True)
        enc.fit(X)
        self.verify_numeric(enc.transform(X_t))

        enc = encoders.OneHotEncoder(verbose=1, return_df=False)
        enc.fit(X)
        self.assertTrue(isinstance(enc.transform(X_t), np.ndarray))

        enc = encoders.OneHotEncoder(verbose=1, return_df=False)
        enc.fit(X)
        self.assertEqual(enc.transform(X_t).shape[1], enc.transform(X_t[X_t['extra'] != 'A']).shape[1], 'We have to get the same count of columns')

        enc = encoders.OneHotEncoder(verbose=1, return_df=True, impute_missing=True)
        enc.fit(X)
        out = enc.transform(X_t_extra)
        self.assertIn('extra_-1', out.columns.values)

        enc = encoders.OneHotEncoder(verbose=1, return_df=True, impute_missing=True, handle_unknown='ignore')
        enc.fit(X)
        out = enc.transform(X_t_extra)
        self.assertEqual(len([x for x in out.columns.values if str(x).startswith('extra_')]), 3)

        enc = encoders.OneHotEncoder(verbose=1, return_df=True, impute_missing=True, handle_unknown='error')
        enc.fit(X)
        with self.assertRaises(ValueError):
            out = enc.transform(X_t_extra)

        enc = encoders.OneHotEncoder(verbose=1, return_df=True, handle_unknown='ignore', use_cat_names=True)
        enc.fit(X)
        out = enc.transform(X_t_extra)
        self.assertIn('extra_A', out.columns.values)

        enc = encoders.OneHotEncoder(verbose=1, return_df=True, use_cat_names=True)
        enc.fit(X)
        out = enc.transform(X_t_extra)
        self.assertIn('extra_-1', out.columns.values)

        # test inverse_transform
        X = self.create_dataset(has_none=False)
        X_t = self.create_dataset(n_rows=100, has_none=False)
        X_t_extra = self.create_dataset(n_rows=100, extras=True, has_none=False)

        enc = encoders.OneHotEncoder(verbose=1)
        enc.fit(X)
        self.verify_numeric(enc.transform(X_t))
        self.verify_inverse_transform(X_t, enc.inverse_transform(enc.transform(X_t)))
        with self.assertRaises(ValueError):
            out = enc.inverse_transform(enc.transform(X_t_extra))

        enc = encoders.OneHotEncoder(verbose=1, use_cat_names=True)
        enc.fit(X)
        self.verify_numeric(enc.transform(X_t))
        self.verify_inverse_transform(X_t, enc.inverse_transform(enc.transform(X_t)))
        with self.assertRaises(ValueError):
            out = enc.inverse_transform(enc.transform(X_t_extra))

    def test_leave_one_out_np(self):
        """

        :return:
        """

        X = self.create_array()
        X_t = self.create_array(n_rows=100)
        y = np.random.randn(X.shape[0])
        y_t = np.random.randn(X_t.shape[0])

        enc = encoders.LeaveOneOutEncoder(verbose=1)
        enc.fit(X, y)
        self.verify_numeric(enc.transform(X_t))
        self.verify_numeric(enc.transform(X_t, y_t))

    def test_leave_one_out(self):


        cols = ['unique_str', 'underscore', 'extra', 'none', 'invariant', 321]
        X = self.create_dataset(n_rows=100)
        X_t = self.create_dataset(n_rows=100, extras=True)
        y = np.random.randn(X.shape[0])
        y_t = np.random.randn(X_t.shape[0])

        # When we use the same LeaveOneOut for two different datasets, it should not explode
        # X_a = pd.DataFrame(data=['1', '1', '1', '2', '2', '2'], columns=['col_a'])
        # X_b = pd.DataFrame(data=['1', '1', '1', '2', '2', '2'], columns=['col_b'])
        # y_dummy = [True, False, True, False, True, False]
        # enc = encoders.LeaveOneOutEncoder()
        # enc.fit(X_a, y_dummy)
        # enc.fit(X_b, y_dummy)
        # self.verify_numeric(enc.transform(X_b))

        enc = encoders.LeaveOneOutEncoder(verbose=1, cols=cols)
        enc.fit(X, y)
        self.verify_numeric(enc.transform(X_t))
        self.verify_numeric(enc.transform(X_t, y_t))

        enc = encoders.LeaveOneOutEncoder(verbose=1)
        enc.fit(X, y)
        self.verify_numeric(enc.transform(X_t))
        self.verify_numeric(enc.transform(X_t, y_t))

        enc = encoders.LeaveOneOutEncoder(verbose=1, drop_invariant=True)
        enc.fit(X, y)
        self.verify_numeric(enc.transform(X_t))
        self.verify_numeric(enc.transform(X_t, y_t))

        enc = encoders.LeaveOneOutEncoder(verbose=1, return_df=False)
        enc.fit(X, y)
        self.assertTrue(isinstance(enc.transform(X_t), np.ndarray))
        self.assertTrue(isinstance(enc.transform(X_t, y_t), np.ndarray))

        enc = encoders.LeaveOneOutEncoder(verbose=1, randomized=True, sigma=0.1)
        enc.fit(X, y)
        self.verify_numeric(enc.transform(X_t))
        self.verify_numeric(enc.transform(X_t, y_t))

        enc = encoders.LeaveOneOutEncoder(impute_missing=True, handle_unknown='error', cols=['underscore'])
        enc.fit(X, y)
        self.assertTrue(np.issubdtype(enc.transform(X_t)['underscore'].dtypes, np.dtype(float)))
        self.assertTrue(np.issubdtype(enc.transform(X_t, y_t)['underscore'].dtypes, np.dtype(float)))

        enc = encoders.LeaveOneOutEncoder(impute_missing=True, handle_unknown='error', cols=['extra'])
        enc.fit(X, y)
        self.assertRaises(ValueError, enc.transform, X_t)
        self.assertRaises(ValueError, enc.transform, (X_t, y_t))

    def test_target_encoder_np(self):
        """

        :return:
        """

        X = self.create_array()
        X_t = self.create_array(n_rows=100)
        y = np.random.randn(X.shape[0])
        y_t = np.random.randn(X_t.shape[0])

        enc = encoders.TargetEncoder(verbose=1)
        enc.fit(X, y)
        self.verify_numeric(enc.transform(X_t))
        self.verify_numeric(enc.transform(X_t, y_t))

    def test_target_encoder(self):
        """

        :return:
        """

        cols = ['unique_str', 'underscore', 'extra', 'none', 'invariant', 321]
        X = self.create_dataset(n_rows=100)
        X_t = self.create_dataset(n_rows=100)
        y = np.random.randn(X.shape[0])
        y_t = np.random.randn(X_t.shape[0])
        #
        # enc = encoders.TargetEncoder(verbose=1, cols=cols)
        # enc.fit(X, y)
        # self.verify_numeric(enc.transform(X_t))
        # self.verify_numeric(enc.transform(X_t, y_t))
        #
        # enc = encoders.TargetEncoder(verbose=1)
        # enc.fit(X, y)
        # self.verify_numeric(enc.transform(X_t))
        # self.verify_numeric(enc.transform(X_t, y_t))
        #
        # enc = encoders.TargetEncoder(verbose=1, drop_invariant=True)
        # enc.fit(X, y)
        # self.verify_numeric(enc.transform(X_t))
        # self.verify_numeric(enc.transform(X_t, y_t))
        #
        # enc = encoders.TargetEncoder(verbose=1, return_df=False)
        # enc.fit(X, y)
        # self.assertTrue(isinstance(enc.transform(X_t), np.ndarray))
        # self.assertTrue(isinstance(enc.transform(X_t, y_t), np.ndarray))

        enc = encoders.TargetEncoder(verbose=1, smoothing=2, min_samples_leaf=2)
        enc.fit(X, y)
        self.verify_numeric(enc.transform(X_t))
        self.verify_numeric(enc.transform(X_t, y_t))

    def test_woe_np(self):
        """

        :return:
        """

        X = self.create_array()
        X_t = self.create_array(n_rows=100)
        y = np.random.randn(X.shape[0]) > 0
        y_t = np.random.randn(X_t.shape[0]) > 0

        enc = encoders.WoeEncoder()
        enc.fit(X, y)
        self.verify_numeric(enc.transform(X_t))
        self.verify_numeric(enc.transform(X_t, y_t))

    def test_woe(self):
        cols = ['unique_str', 'underscore', 'extra', 'none', 'invariant', 321]
        X = self.create_dataset(n_rows=100)
        X_t = self.create_dataset(n_rows=100, extras=True)
        y = np.random.randn(X.shape[0]) > 0
        y_t = np.random.randn(X_t.shape[0]) > 0

        # Balanced label with balanced features
        X_balanced = pd.DataFrame(data=['1', '1', '1', '2', '2', '2'], columns=['col1'])
        y_balanced = [True, False, True, False, True, False]
        enc = encoders.WoeEncoder()
        enc.fit(X_balanced, y_balanced)
        X1 = enc.transform(X_balanced)
        self.assertTrue(all(X1.sum() < 0.001), "When the class label is balanced, WoE should sum to 0 in each transformed column")

        enc = encoders.WoeEncoder(cols=cols)
        enc.fit(X, y)
        X1 = enc.transform(X_t)
        self.verify_numeric(X1[cols])
        self.assertTrue(np.isfinite(X1[cols].values).all(), 'There must not be any NaN, inf or -inf in the transformed columns')
        self.assertEqual(len(list(X_t)), len(list(X1)), 'The count of attributes must not change')
        self.assertEqual(len(X_t), len(X1), 'The count of rows must not change')
        X2 = enc.transform(X_t, y_t)
        self.verify_numeric(X2)
        self.assertTrue(np.isfinite(X2[cols].values).all(), 'There must not be any NaN, inf or -inf in the transformed columns')
        self.assertEqual(len(list(X_t)), len(list(X2)), 'The count of attributes must not change')
        self.assertEqual(len(X_t), len(X2), 'The count of rows must not change')
        X3 = enc.transform(X, y)
        self.verify_numeric(X3)
        self.assertTrue(np.isfinite(X3[cols].values).all(), 'There must not be any NaN, inf or -inf in the transformed columns')
        self.assertEqual(len(list(X)), len(list(X3)), 'The count of attributes must not change')
        self.assertEqual(len(X), len(X3), 'The count of rows must not change')
        self.assertTrue(X3['unique_str'].var() < 0.001, 'The unique string column must not be predictive of the label')
        X4 = enc.fit_transform(X, y)
        self.verify_numeric(X4)
        self.assertTrue(np.isfinite(X4[cols].values).all(), 'There must not be any NaN, inf or -inf in the transformed columns')
        self.assertEqual(len(list(X)), len(list(X4)), 'The count of attributes must not change')
        self.assertEqual(len(X), len(X4), 'The count of rows must not change')
        self.assertTrue(X4['unique_str'].var() < 0.001, 'The unique string column must not be predictive of the label')

        enc = encoders.WoeEncoder()
        enc.fit(X, y)
        X1 = enc.transform(X_t)
        self.assertEqual(len(list(X)), len(list(X1)), 'The count of attributes must not change')
        self.assertEqual(len(X), len(X1), 'The count of rows must not change')
        self.verify_numeric(X1)
        X2 = enc.transform(X_t, y_t)
        self.verify_numeric(X2)
        self.assertEqual(len(list(X_t)), len(list(X2)), 'The count of attributes must not change')
        self.assertEqual(len(X_t), len(X2), 'The count of rows must not change')

        enc = encoders.WoeEncoder(drop_invariant=True)
        enc.fit(X, y)
        self.verify_numeric(enc.transform(X_t))
        self.verify_numeric(enc.transform(X_t, y_t))

        # Numpy
        enc = encoders.WoeEncoder(return_df=False)
        enc.fit(X, y)
        self.assertTrue(isinstance(enc.transform(X_t), np.ndarray))
        self.assertTrue(isinstance(enc.transform(X_t, y_t), np.ndarray))

        # Seed
        enc = encoders.WoeEncoder(cols=cols, random_state=2001, randomized=True)
        enc.fit(X, y)
        X1 = enc.transform(X_t, y_t)
        X2 = enc.transform(X_t, y_t)
        self.assertTrue(X1.equals(X2), "When the seed is given, the results must be identical")
        self.verify_numeric(X1)
        self.verify_numeric(X2)

        ## Exceptions
        enc = encoders.WoeEncoder()
        self.assertRaises(ValueError, enc.transform, X, 'We have to first train the encoder...')

        enc = encoders.WoeEncoder()
        enc.fit(X, y)
        self.assertRaises(ValueError, enc.transform, X_t.iloc[:,0:3], 'Wrong count of attributes...')

        y_invariant = [True, True, True, True, True, True]
        enc = encoders.WoeEncoder()
        with self.assertRaises(AssertionError):
            enc.fit(X_balanced, y_invariant)

        # Branch coverage unit tests - no cols
        enc = encoders.WoeEncoder(cols=[])
        enc.fit(X, y)
        self.assertTrue(enc.transform(X_t).equals(X_t))

        # Missing values in the label
        y_missing = [True, False, None, True, True, True]
        enc = encoders.WoeEncoder()
        with self.assertRaises(AssertionError):
            enc.fit(X_balanced, y_missing)

        # Impute missing
        enc = encoders.WoeEncoder(impute_missing=False)
        enc.fit(X, y)
        X1 = enc.transform(X_t)
        self.verify_numeric(X1)
        self.assertTrue(X1.isnull().values.any())
        self.assertEqual(len(list(X)), len(list(X1)), 'The count of attributes must not change')
        self.assertEqual(len(X), len(X1), 'The count of rows must not change')
        X2 = enc.transform(X_t, y_t)
        self.verify_numeric(X2)
        self.assertTrue(X1.isnull().values.any())
        self.assertEqual(len(list(X_t)), len(list(X2)), 'The count of attributes must not change')
        self.assertEqual(len(X_t), len(X2), 'The count of rows must not change')

        enc = encoders.WoeEncoder(impute_missing=True, handle_unknown='error')
        enc.fit(X, y)
        with self.assertRaises(ValueError):
            self.verify_numeric(enc.transform(X_t))
        with self.assertRaises(ValueError):
            self.verify_numeric(enc.transform(X_t, y_t))

    def test_doc(self):
        suite = unittest.TestSuite()

        for filename in os.listdir('../'):
            if filename.endswith(".py"):
                suite.addTest(doctest.DocFileSuite('../' + filename))

        runner = unittest.TextTestRunner(verbosity=2)
        runner.run(suite)


    def test_fit_HaveConstructorSetSmoothingAndMinSamplesLeaf_ExpectUsedInFit(self):
        """
        :return:
        """
        k = 2
        f = 10
        binary_cat_example = pd.DataFrame(
            {'Trend': ['UP', 'UP', 'DOWN', 'FLAT', 'DOWN', 'UP', 'DOWN', 'FLAT', 'FLAT', 'FLAT'],
             'target': [1, 1, 0, 0, 1, 0, 0, 0, 1, 1]})
        encoder = encoders.TargetEncoder(cols=['Trend'], min_samples_leaf=k, smoothing=f)
        encoder.fit(binary_cat_example, binary_cat_example['target'])
        trend_mapping = encoder.mapping[0]['mapping']
        self.assertAlmostEquals(0.4125, trend_mapping['DOWN']['smoothing'], delta=1e-4)
        self.assertEqual(0.5, trend_mapping['FLAT']['smoothing'])
        self.assertAlmostEquals(0.5874, trend_mapping['UP']['smoothing'], delta=1e-4)

    def test_fit_transform_HaveConstructorSetSmoothingAndMinSamplesLeaf_ExpectCorrectValueInResult(self):
        """
        :return:
        """
        k = 2
        f = 10
        binary_cat_example = pd.DataFrame(
            {'Trend': ['UP', 'UP', 'DOWN', 'FLAT', 'DOWN', 'UP', 'DOWN', 'FLAT', 'FLAT', 'FLAT'],
             'target': [1, 1, 0, 0, 1, 0, 0, 0, 1, 1]})
        encoder = encoders.TargetEncoder(cols=['Trend'], min_samples_leaf=k, smoothing=f)
        result = encoder.fit_transform(binary_cat_example, binary_cat_example['target'])
        values = result['Trend'].values
        self.assertAlmostEquals(0.5874, values[0], delta=1e-4)
        self.assertAlmostEquals(0.5874, values[1], delta=1e-4)
        self.assertAlmostEquals(0.4125, values[2], delta=1e-4)
        self.assertEqual(0.5, values[3])