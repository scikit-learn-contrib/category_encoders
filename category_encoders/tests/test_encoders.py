import unittest
import random
import pandas as pd
import category_encoders as encoders
import numpy as np

__author__ = 'willmcginnis'


class TestEncoders(unittest.TestCase):
    """
    """

    def verify_numeric(self, X_test):
        for dt in X_test.dtypes:
            numeric = False
            if np.issubdtype(dt, int) or np.issubdtype(dt, float):
                numeric = True
            self.assertTrue(numeric)

    def create_dataset(self, n_rows=1000, extras=False):
        """
        Creates a dataset with some categorical variables
        :return:
        """

        ds = [[
            random.random(),
            random.random(),
            random.choice(['A', 'B', 'C']),
            random.choice(['A', 'B', 'C', 'D']) if extras else random.choice(['A', 'B', 'C']),
            random.choice(['A', 'B', 'C', None]),
            random.choice(['A', 'B', 'C'])
        ] for _ in range(n_rows)]

        df = pd.DataFrame(ds, columns=['A', 'B', 'C1', 'D', 'E', 'F'])
        return df

    def create_array(self, n_rows=1000, extras=False):
        """
        Creates a dataset with some categorical variables
        :return:
        """

        ds = [[
            random.random(),
            random.random(),
            random.choice(['A', 'B', 'C']),
            random.choice(['A', 'B', 'C', 'D']) if extras else random.choice(['A', 'B', 'C']),
            random.choice(['A', 'B', 'C', None]),
            random.choice(['A', 'B', 'C'])
        ] for _ in range(n_rows)]

        return np.array(ds)

    def test_hashing_np(self):
        """
        Creates a dataset and encodes with with the hashing trick

        :return:
        """

        X = self.create_array(n_rows=1000)
        X_t = self.create_array(n_rows=100)

        enc = encoders.HashingEncoder(verbose=1, n_components=128)
        enc.fit(X, None)
        self.verify_numeric(enc.transform(X_t))

    def test_hashing(self):
        """
        Creates a dataset and encodes with with the hashing trick

        :return:
        """

        cols = ['C1', 'D', 'E', 'F']
        X = self.create_dataset(n_rows=1000)
        X_t = self.create_dataset(n_rows=100)

        enc = encoders.HashingEncoder(verbose=1, n_components=128, cols=cols)
        enc.fit(X, None)
        self.verify_numeric(enc.transform(X_t))

        enc = encoders.HashingEncoder(verbose=1, n_components=32)
        enc.fit(X, None)
        self.verify_numeric(enc.transform(X_t))

        enc = encoders.HashingEncoder(verbose=1, n_components=32, drop_invariant=True)
        enc.fit(X, None)
        self.verify_numeric(enc.transform(X_t))

        enc = encoders.HashingEncoder(verbose=1, n_components=32, return_df=False)
        enc.fit(X, None)
        self.assertTrue(isinstance(enc.transform(X_t), np.ndarray))

    def test_ordinal_np(self):
        """

        :return:
        """

        X = self.create_array(n_rows=1000)
        X_t = self.create_array(n_rows=100)

        enc = encoders.OrdinalEncoder(verbose=1)
        enc.fit(X, None)
        self.verify_numeric(enc.transform(X_t))

    def test_ordinal(self):
        """

        :return:
        """

        cols = ['C1', 'D', 'E', 'F']
        X = self.create_dataset(n_rows=1000)
        X_t = self.create_dataset(n_rows=100)
        X_t_extra = self.create_dataset(n_rows=100, extras=True)

        enc = encoders.OrdinalEncoder(verbose=1, cols=cols)
        enc.fit(X, None)
        self.verify_numeric(enc.transform(X_t))

        enc = encoders.OrdinalEncoder(verbose=1)
        enc.fit(X, None)
        self.verify_numeric(enc.transform(X_t))

        enc = encoders.OrdinalEncoder(verbose=1, drop_invariant=True)
        enc.fit(X, None)
        self.verify_numeric(enc.transform(X_t))

        enc = encoders.OrdinalEncoder(verbose=1, return_df=False)
        enc.fit(X, None)
        self.assertTrue(isinstance(enc.transform(X_t), np.ndarray))

        enc = encoders.OrdinalEncoder(verbose=1, return_df=True, impute_missing=True, handle_unknown='impute')
        enc.fit(X, None)
        out = enc.transform(X_t_extra)
        self.assertEqual(len(set(out['D'].values)), 4)
        self.assertIn(-1, set(out['D'].values))
        self.assertFalse(enc.mapping is None)

        enc = encoders.OrdinalEncoder(verbose=1, return_df=True, impute_missing=True, handle_unknown='ignore')
        enc.fit(X, None)
        out = enc.transform(X_t_extra)
        out_cats = [x for x in set(out['D'].values) if np.isfinite(x)]
        self.assertEqual(len(out_cats), 3)
        self.assertFalse(enc.mapping is None)

        enc = encoders.OrdinalEncoder(verbose=1, return_df=True, impute_missing=True, handle_unknown='error')
        enc.fit(X, None)
        with self.assertRaises(ValueError):
            out = enc.transform(X_t_extra)

    def test_backward_difference_np(self):
        """

        :return:
        """

        X = self.create_array(n_rows=1000)
        X_t = self.create_array(n_rows=100)

        enc = encoders.BackwardDifferenceEncoder(verbose=1)
        enc.fit(X, None)
        self.verify_numeric(enc.transform(X_t))

    def test_backward_difference(self):
        """

        :return:
        """

        cols = ['C1', 'D', 'E', 'F']
        X = self.create_dataset(n_rows=1000)
        X_t = self.create_dataset(n_rows=100)

        enc = encoders.BackwardDifferenceEncoder(verbose=1, cols=cols)
        enc.fit(X, None)
        self.verify_numeric(enc.transform(X_t))

        enc = encoders.BackwardDifferenceEncoder(verbose=1)
        enc.fit(X, None)
        self.verify_numeric(enc.transform(X_t))

        enc = encoders.BackwardDifferenceEncoder(verbose=1, drop_invariant=True)
        enc.fit(X, None)
        self.verify_numeric(enc.transform(X_t))

        enc = encoders.BackwardDifferenceEncoder(verbose=1, return_df=False)
        enc.fit(X, None)
        self.assertTrue(isinstance(enc.transform(X_t), np.ndarray))

    def test_binary_np(self):
        """

        :return:
        """

        X = self.create_array(n_rows=1000)
        X_t = self.create_array(n_rows=100)

        enc = encoders.BinaryEncoder(verbose=1)
        enc.fit(X, None)
        self.verify_numeric(enc.transform(X_t))

    def test_binary(self):
        """

        :return:
        """

        cols = ['C1', 'D', 'E', 'F']
        X = self.create_dataset(n_rows=1000)
        X_t = self.create_dataset(n_rows=100)

        enc = encoders.BinaryEncoder(verbose=1, cols=cols)
        enc.fit(X, None)
        self.verify_numeric(enc.transform(X_t))

        enc = encoders.BinaryEncoder(verbose=1)
        enc.fit(X, None)
        self.verify_numeric(enc.transform(X_t))

        enc = encoders.BinaryEncoder(verbose=1, drop_invariant=True)
        enc.fit(X, None)
        self.verify_numeric(enc.transform(X_t))

        enc = encoders.BinaryEncoder(verbose=1, return_df=False)
        enc.fit(X, None)
        self.assertTrue(isinstance(enc.transform(X_t), np.ndarray))

    def test_basen_np(self):
        """

        :return:
        """

        X = self.create_array(n_rows=1000)
        X_t = self.create_array(n_rows=100)

        enc = encoders.BaseNEncoder(verbose=1)
        enc.fit(X, None)
        self.verify_numeric(enc.transform(X_t))

    def test_basen(self):
        """

        :return:
        """

        cols = ['C1', 'D', 'E', 'F']
        X = self.create_dataset(n_rows=1000)
        X_t = self.create_dataset(n_rows=100)

        enc = encoders.BaseNEncoder(verbose=1, cols=cols)
        enc.fit(X, None)
        self.verify_numeric(enc.transform(X_t))

        enc = encoders.BaseNEncoder(verbose=1)
        enc.fit(X, None)
        self.verify_numeric(enc.transform(X_t))

        enc = encoders.BaseNEncoder(verbose=1, drop_invariant=True)
        enc.fit(X, None)
        self.verify_numeric(enc.transform(X_t))

        enc = encoders.BaseNEncoder(verbose=1, return_df=False)
        enc.fit(X, None)
        self.assertTrue(isinstance(enc.transform(X_t), np.ndarray))

    def test_helmert_np(self):
        """

        :return:
        """

        X = self.create_array(n_rows=1000)
        X_t = self.create_array(n_rows=100)

        enc = encoders.HelmertEncoder(verbose=1)
        enc.fit(X, None)
        self.verify_numeric(enc.transform(X_t))

    def test_helmert(self):
        """

        :return:
        """

        cols = ['C1', 'D', 'E', 'F']
        X = self.create_dataset(n_rows=1000)
        X_t = self.create_dataset(n_rows=100)

        enc = encoders.HelmertEncoder(verbose=1, cols=cols)
        enc.fit(X, None)
        self.verify_numeric(enc.transform(X_t))

        enc = encoders.HelmertEncoder(verbose=1)
        enc.fit(X, None)
        self.verify_numeric(enc.transform(X_t))

        enc = encoders.HelmertEncoder(verbose=1, drop_invariant=True)
        enc.fit(X, None)
        self.verify_numeric(enc.transform(X_t))

        enc = encoders.HelmertEncoder(verbose=1, return_df=False)
        enc.fit(X, None)
        self.assertTrue(isinstance(enc.transform(X_t), np.ndarray))

    def test_polynomial_np(self):
        """

        :return:
        """

        X = self.create_array(n_rows=1000)
        X_t = self.create_array(n_rows=100)

        enc = encoders.PolynomialEncoder(verbose=1)
        enc.fit(X, None)
        self.verify_numeric(enc.transform(X_t))

    def test_polynomial(self):
        """

        :return:
        """

        cols = ['C1', 'D', 'E', 'F']
        X = self.create_dataset(n_rows=1000)
        X_t = self.create_dataset(n_rows=100)

        enc = encoders.PolynomialEncoder(verbose=1, cols=cols)
        enc.fit(X, None)
        self.verify_numeric(enc.transform(X_t))

        enc = encoders.PolynomialEncoder(verbose=1)
        enc.fit(X, None)
        self.verify_numeric(enc.transform(X_t))

        enc = encoders.PolynomialEncoder(verbose=1, drop_invariant=True)
        enc.fit(X, None)
        self.verify_numeric(enc.transform(X_t))

        enc = encoders.PolynomialEncoder(verbose=1, return_df=False)
        enc.fit(X, None)
        self.assertTrue(isinstance(enc.transform(X_t), np.ndarray))

    def test_sum_np(self):
        """

        :return:
        """

        X = self.create_array(n_rows=1000)
        X_t = self.create_array(n_rows=100)

        enc = encoders.SumEncoder(verbose=1)
        enc.fit(X, None)
        self.verify_numeric(enc.transform(X_t))

    def test_sum(self):
        """

        :return:
        """

        cols = ['C1', 'D', 'E', 'F']
        X = self.create_dataset(n_rows=1000)
        X_t = self.create_dataset(n_rows=100)

        enc = encoders.SumEncoder(verbose=1, cols=cols)
        enc.fit(X, None)
        self.verify_numeric(enc.transform(X_t))

        enc = encoders.SumEncoder(verbose=1)
        enc.fit(X, None)
        self.verify_numeric(enc.transform(X_t))

        enc = encoders.SumEncoder(verbose=1, drop_invariant=True)
        enc.fit(X, None)
        self.verify_numeric(enc.transform(X_t))

        enc = encoders.SumEncoder(verbose=1, return_df=False)
        enc.fit(X, None)
        self.assertTrue(isinstance(enc.transform(X_t), np.ndarray))

    def test_onehot_np(self):
        """

        :return:
        """

        X = self.create_array(n_rows=1000)
        X_t = self.create_array(n_rows=100)

        enc = encoders.OneHotEncoder(verbose=1)
        enc.fit(X, None)
        self.verify_numeric(enc.transform(X_t))

    def test_onehot(self):
        """

        :return:
        """

        cols = ['C1', 'D', 'E', 'F']
        X = self.create_dataset(n_rows=1000)
        X_t = self.create_dataset(n_rows=100)
        X_t_extra = self.create_dataset(n_rows=100, extras=True)

        enc = encoders.OneHotEncoder(verbose=1, cols=cols)
        enc.fit(X, None)
        self.verify_numeric(enc.transform(X_t))

        enc = encoders.OneHotEncoder(verbose=1)
        enc.fit(X, None)
        self.verify_numeric(enc.transform(X_t))

        enc = encoders.OneHotEncoder(verbose=1, drop_invariant=True)
        enc.fit(X, None)
        self.verify_numeric(enc.transform(X_t))

        enc = encoders.OneHotEncoder(verbose=1, return_df=False)
        enc.fit(X, None)
        self.assertTrue(isinstance(enc.transform(X_t), np.ndarray))

        enc = encoders.OneHotEncoder(verbose=1, return_df=True, impute_missing=True, handle_unknown='impute')
        enc.fit(X, None)
        out = enc.transform(X_t_extra)
        self.assertIn('D_-1', out.columns.values)

        enc = encoders.OneHotEncoder(verbose=1, return_df=True, impute_missing=True, handle_unknown='ignore')
        enc.fit(X, None)
        out = enc.transform(X_t_extra)
        self.assertEqual(len([x for x in out.columns.values if x.startswith('D_')]), 3)

        enc = encoders.OneHotEncoder(verbose=1, return_df=True, impute_missing=True, handle_unknown='error')
        enc.fit(X, None)
        with self.assertRaises(ValueError):
            out = enc.transform(X_t_extra)
