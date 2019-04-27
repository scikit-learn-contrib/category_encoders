from unittest2 import TestCase  # or `from unittest import ...` if on Python 3.4+
import numpy as np

import category_encoders as encoders


class TestJamesSteinEncoder(TestCase):

    def test_small_samples_independent(self):
        X = np.array(['a', 'b', 'b'])
        y = np.array([1, 0, 1])
        out = encoders.JamesSteinEncoder(return_df=False, model='independent').fit_transform(X, y)
        self.assertEqual([1, 0.5, 0.5],
                          list(out),
                          'When the count of unique values in the column is <4 (here it is 2), James-Stein estimator returns (unbiased) sample means')

    def test_large_samples(self):
        X = np.array(['a', 'b', 'b', 'c', 'd'])
        y = np.array([1, 0, 1, 0, 0])
        out = encoders.JamesSteinEncoder(return_df=False, model='independent').fit_transform(X, y)
        self.assertNotEqual([1, 0.5, 0.5, 0, 0],
                          list(out),
                          'Shrinkage should kick in with 4 or more unique values')
        self.assertTrue(np.max(out) <= 1, 'This should still be a probability')
        self.assertTrue(np.min(out) >= 0, 'This should still be a probability')

    def test_zero_variance(self):
        X = np.array(['a', 'b', 'c', 'd', 'd'])
        y = np.array([0, 1, 1, 1, 1])
        out = encoders.JamesSteinEncoder(return_df=False, model='independent').fit_transform(X, y)
        self.assertEqual([0, 1, 1, 1, 1],
                          list(out),
                          'Should not result into division by zero')

    def test_continuous_target(self):
        X = np.array(['a', 'b', 'b', 'c'])
        y = np.array([-10, 0, 0, 10])
        out = encoders.JamesSteinEncoder(return_df=False, model='independent').fit_transform(X, y)
        self.assertEqual([-10, 0, 0, 10],
                          list(out),
                          'The model assumes normal distribution -> we support real numbers')

    # Pooled
    def test_continuous_target_pooled(self):
        X = np.array(['a', 'b', 'b', 'c'])
        y = np.array([-10, 0, 0, 10])
        out = encoders.JamesSteinEncoder(return_df=False, model='pooled').fit_transform(X, y)
        self.assertEqual([-10, 0, 0, 10],
                          list(out),
                          'The model assumes normal distribution -> we support real numbers')

    def test_large_samples_pooled(self):
        X = np.array(['a', 'b', 'b', 'c', 'd'])
        y = np.array([1, 0, 1, 0, 0])
        out = encoders.JamesSteinEncoder(return_df=False, model='pooled').fit_transform(X, y)
        self.assertNotEqual([1, 0.5, 0.5, 0, 0],
                          list(out),
                          'Shrinkage should kick in with 4 or more unique values')
        self.assertTrue(np.max(out) <= 1, 'This should still be a probability')
        self.assertTrue(np.min(out) >= 0, 'This should still be a probability')

    def test_ids_small_pooled(self):
        X = np.array(['a', 'b', 'c'])
        y = np.array([1, 0, 1])
        out = encoders.JamesSteinEncoder(model='pooled').fit_transform(X, y)
        self.assertTrue(all(np.var(out) == 0),
                          'This is not a standard behaviour of James-Stein estimator. But it helps a lot if we treat id-like attributes as non-predictive.')

    def test_ids_large_pooled(self):
        X = np.array(['a', 'b', 'c', 'd', 'e'])
        y = np.array([1, 0, 1, 0, 1])
        out = encoders.JamesSteinEncoder(model='pooled').fit_transform(X, y)
        self.assertTrue(all(np.var(out) == 0),
                          'This is not a standard behaviour of James-Stein estimator. But it helps a lot if we treat id-like attributes as non-predictive.')

    # Beta
    def test_continuous_target_beta(self):
        X = np.array(['a', 'b', 'b', 'c'])
        y = np.array([-10, 0, 0, 10])
        out = encoders.JamesSteinEncoder(return_df=False, model='beta').fit_transform(X, y)
        self.assertEqual([-2, 0, 0, 2],
                         list(out),
                         'The model assumes normal distribution -> we support real numbers')

    def test_large_samples_beta(self):
        X = np.array(['a', 'b', 'b', 'c', 'd'])
        y = np.array([1, 0, 1, 0, 0])
        out = encoders.JamesSteinEncoder(return_df=False, model='beta').fit_transform(X, y)
        self.assertNotEqual([1, 0.5, 0.5, 0, 0],
                            list(out),
                            'Shrinkage should kick in with 4 or more unique values')
        self.assertTrue(np.max(out) <= 1, 'This should still be a probability')
        self.assertTrue(np.min(out) >= 0, 'This should still be a probability')

    def test_ids_small_beta(self):
        X = np.array(['a', 'b', 'c'])
        y = np.array([1, 0, 1])
        out = encoders.JamesSteinEncoder(model='beta').fit_transform(X, y)
        self.assertTrue(all(np.var(out) == 0),
                        'This is not a standard behaviour of James-Stein estimator. But it helps a lot if we treat id-like attributes as non-predictive.')

    def test_ids_large_beta(self):
        X = np.array(['a', 'b', 'c', 'd', 'e'])
        y = np.array([1, 0, 1, 0, 1])
        out = encoders.JamesSteinEncoder(model='beta').fit_transform(X, y)
        self.assertTrue(all(np.var(out) == 0),
                        'This is not a standard behaviour of James-Stein estimator. But it helps a lot if we treat id-like attributes as non-predictive.')

    # Binary
    def test_small_samples_binary(self):
        X = np.array(['a', 'b', 'b'])
        y = np.array([1, 0, 1])
        out = encoders.JamesSteinEncoder(return_df=False, model='binary').fit_transform(X, y)
        self.assertTrue(np.sum(np.abs([np.log((1.5*1.5)/(0.5*1.5)), np.log((0.5*1.5)/(1.5*1.5)), np.log((0.5*1.5)/(1.5*1.5))] - np.transpose(out))) < 0.001,
                          'When the count of unique values in the column is <4 (here it is 2), James-Stein estimator returns (unbiased) sample means')

    def test_large_samples_binary(self):
        X = np.array(['a', 'b', 'b', 'c', 'd'])
        y = np.array([1, 0, 1, 0, 0])
        out = encoders.JamesSteinEncoder(return_df=False, model='binary').fit_transform(X, y)
        self.assertNotEqual([1, 0.5, 0.5, 0, 0],
                          list(out),
                          'Shrinkage should kick in with 4 or more unique values')

    def test_identifiers_small_binary(self):
        X = np.array(['a', 'b', 'c'])
        y = np.array([1, 0, 1])
        out = encoders.JamesSteinEncoder(model='binary').fit_transform(X, y)
        self.assertTrue(all(np.var(out) == 0),
                          'This is not a standard behaviour of James-Stein estimator. But it helps a lot if we treat id-like attributes as non-predictive.')

    def test_identifiers_large_binary(self):
        X = np.array(['a', 'b', 'c', 'd', 'e'])
        y = np.array([1, 0, 1, 0, 1])
        out = encoders.JamesSteinEncoder(model='binary').fit_transform(X, y)
        self.assertTrue(all(np.var(out) == 0),
                          'This is not a standard behaviour of James-Stein estimator. But it helps a lot if we treat id-like attributes as non-predictive.')
