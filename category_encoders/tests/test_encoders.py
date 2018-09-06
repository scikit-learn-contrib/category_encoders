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

    def verify_inverse_transform(self, x, x_inv):
        """
        Verify x is equal to x_inv.

        """
        is_inv = False
        if (x.columns == x_inv.columns).all() and x.shape == x_inv.shape:
            is_inv = (x == x_inv).all(1).all(0)
        self.assertTrue(is_inv)

    def create_dataset(self, n_rows=1000, extras=False, has_none=True):
        """
        Creates a dataset with some categorical variables
        :return:
        """

        ds = [[
                  random.random(),
                  random.random(),
                  random.choice(['A', 'B', 'C']),
                  random.choice(['A', 'B', 'C', 'D']) if extras else random.choice(['A', 'B', 'C']),
                  random.choice(['A', 'B', 'C', None]) if has_none else random.choice(['A', 'B', 'C']),
                  random.choice(['A', 'B', 'C'])
              ] for _ in range(n_rows)]

        df = pd.DataFrame(ds, columns=['A', 'B', 'C1', 'D', 'E', 'F'])
        return df

    def create_array(self, n_rows=1000, extras=False, has_none=True):
        """
        Creates a dataset with some categorical variables
        :return:
        """

        ds = [[
                  random.random(),
                  random.random(),
                  random.choice(['A', 'B', 'C']),
                  random.choice(['A', 'B', 'C', 'D']) if extras else random.choice(['A', 'B', 'C']),
                  random.choice(['A', 'B', 'C', None]) if has_none else random.choice(['A', 'B', 'C']),
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
        self.assertIn(0, set(out['D'].values))
        self.assertFalse(enc.mapping is None)
        self.assertTrue(len(enc.mapping) > 0)

        enc = encoders.OrdinalEncoder(verbose=1, mapping=enc.mapping, return_df=True, impute_missing=True,
                                      handle_unknown='impute')
        enc.fit(X, None)
        out = enc.transform(X_t_extra)
        self.assertEqual(len(set(out['D'].values)), 4)
        self.assertIn(0, set(out['D'].values))
        self.assertTrue(len(enc.mapping) > 0)

        enc = encoders.OrdinalEncoder(verbose=1, return_df=True, impute_missing=True, handle_unknown='ignore')
        enc.fit(X, None)
        out = enc.transform(X_t_extra)
        out_cats = [x for x in set(out['D'].values) if np.isfinite(x)]
        self.assertEqual(len(out_cats), 3)
        self.assertFalse(enc.mapping is None)

        enc = encoders.OrdinalEncoder(verbose=1, return_df=True, handle_unknown='error')
        enc.fit(X, None)
        with self.assertRaises(ValueError):
            out = enc.transform(X_t_extra)

        # test inverse_transform
        X = self.create_dataset(n_rows=1000, has_none=False)
        X_t = self.create_dataset(n_rows=100, has_none=False)
        X_t_extra = self.create_dataset(n_rows=100, extras=True, has_none=False)

        enc = encoders.OrdinalEncoder(verbose=1)
        enc.fit(X, None)
        self.verify_numeric(enc.transform(X_t))
        self.verify_inverse_transform(X_t, enc.inverse_transform(enc.transform(X_t)))
        with self.assertRaises(ValueError):
            out = enc.inverse_transform(enc.transform(X_t_extra))

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

        # test inverse_transform
        X = self.create_dataset(n_rows=1000, has_none=False)
        X_t = self.create_dataset(n_rows=100, has_none=False)
        X_t_extra = self.create_dataset(n_rows=100, extras=True, has_none=False)

        enc = encoders.BinaryEncoder(verbose=1)
        enc.fit(X, None)
        self.verify_numeric(enc.transform(X_t))
        self.verify_inverse_transform(X_t, enc.inverse_transform(enc.transform(X_t)))
        with self.assertRaises(ValueError):
            out = enc.inverse_transform(enc.transform(X_t_extra))

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
        X_t_extra = self.create_dataset(n_rows=100, extras=True)

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

        # test inverse_transform
        X = self.create_dataset(n_rows=1000, has_none=False)
        X_t = self.create_dataset(n_rows=100, has_none=False)
        X_t_extra = self.create_dataset(n_rows=100, extras=True, has_none=False)

        enc = encoders.BaseNEncoder(verbose=1)
        enc.fit(X, None)
        self.verify_numeric(enc.transform(X_t))
        self.verify_inverse_transform(X_t, enc.inverse_transform(enc.transform(X_t)))

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

        enc = encoders.OneHotEncoder(verbose=1, return_df=False)
        enc.fit(X, None)
        self.assertTrue(enc.transform(X_t[X_t['D'] != 'A']).shape[1] == 18)

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

        enc = encoders.OneHotEncoder(verbose=1, return_df=True, handle_unknown='ignore', use_cat_names=True)
        enc.fit(X, None)
        out = enc.transform(X_t_extra)
        self.assertIn('D_A', out.columns.values)

        enc = encoders.OneHotEncoder(verbose=1, return_df=True, handle_unknown='impute', use_cat_names=True)
        enc.fit(X, None)
        out = enc.transform(X_t_extra)
        self.assertIn('D_-1', out.columns.values)

        # test inverse_transform
        X = self.create_dataset(n_rows=1000, has_none=False)
        X_t = self.create_dataset(n_rows=100, has_none=False)
        X_t_extra = self.create_dataset(n_rows=100, extras=True, has_none=False)

        enc = encoders.OneHotEncoder(verbose=1)
        enc.fit(X, None)
        self.verify_numeric(enc.transform(X_t))
        self.verify_inverse_transform(X_t, enc.inverse_transform(enc.transform(X_t)))
        with self.assertRaises(ValueError):
            out = enc.inverse_transform(enc.transform(X_t_extra))

    def test_leave_one_out_np(self):
        """

        :return:
        """

        X = self.create_array(n_rows=1000)
        X_t = self.create_array(n_rows=100)
        y = np.random.randn(X.shape[0])
        y_t = np.random.randn(X_t.shape[0])

        enc = encoders.LeaveOneOutEncoder(verbose=1)
        enc.fit(X, y)
        self.verify_numeric(enc.transform(X_t))
        self.verify_numeric(enc.transform(X_t, y_t))

    def test_leave_one_out(self):
        """

        :return:
        """

        cols = ['C1', 'D', 'E', 'F']
        X = self.create_dataset(n_rows=1000)
        X_t = self.create_dataset(n_rows=100)
        y = np.random.randn(X.shape[0])
        y_t = np.random.randn(X_t.shape[0])

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

    def test_fit_callTwiceOnDifferentData_ExpectRefit(self):
        x_a = pd.DataFrame(data=['1', '1', '1', '2', '2', '2'], columns=['col_a'])
        x_b = pd.DataFrame(data=['1', '1', '1', '2', '2', '2'], columns=['col_b'])  # Different column name
        y_dummy = [True, False, True, False, True, False]
        encoder = encoders.LeaveOneOutEncoder()

        encoder.fit(x_a, y_dummy)
        encoder.fit(x_b, y_dummy)
        mapping = encoder.mapping

        self.assertEqual(1, len(mapping))
        col_b_mapping = mapping[0]
        self.assertEqual('col_b', col_b_mapping['col'])
        self.assertEqual({'sum': 2.0, 'count': 3, 'mean': 2.0/3.0}, col_b_mapping['mapping']['1'])
        self.assertEqual({'sum': 1.0, 'count': 3, 'mean': 01.0/3.0}, col_b_mapping['mapping']['2'])

    def test_target_encode_np(self):
        """

        :return:
        """

        X = self.create_array(n_rows=1000)
        X_t = self.create_array(n_rows=100)
        y = np.random.randn(X.shape[0])
        y_t = np.random.randn(X_t.shape[0])

        enc = encoders.TargetEncoder(verbose=1)
        enc.fit(X, y)
        self.verify_numeric(enc.transform(X_t))
        self.verify_numeric(enc.transform(X_t, y_t))

    def test_target_encode_out(self):
        """

        :return:
        """

        cols = ['C1', 'D', 'E', 'F']
        X = self.create_dataset(n_rows=1000)
        X_t = self.create_dataset(n_rows=100)
        y = np.random.randn(X.shape[0])
        y_t = np.random.randn(X_t.shape[0])

        enc = encoders.TargetEncoder(verbose=1, cols=cols)
        enc.fit(X, y)
        self.verify_numeric(enc.transform(X_t))
        self.verify_numeric(enc.transform(X_t, y_t))

        enc = encoders.TargetEncoder(verbose=1)
        enc.fit(X, y)
        self.verify_numeric(enc.transform(X_t))
        self.verify_numeric(enc.transform(X_t, y_t))

        enc = encoders.TargetEncoder(verbose=1, drop_invariant=True)
        enc.fit(X, y)
        self.verify_numeric(enc.transform(X_t))
        self.verify_numeric(enc.transform(X_t, y_t))

        enc = encoders.TargetEncoder(verbose=1, return_df=False)
        enc.fit(X, y)
        self.assertTrue(isinstance(enc.transform(X_t), np.ndarray))
        self.assertTrue(isinstance(enc.transform(X_t, y_t), np.ndarray))

        enc = encoders.TargetEncoder(verbose=1, smoothing=2, min_samples_leaf=2)
        enc.fit(X, y)
        self.verify_numeric(enc.transform(X_t))
        self.verify_numeric(enc.transform(X_t, y_t))

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
