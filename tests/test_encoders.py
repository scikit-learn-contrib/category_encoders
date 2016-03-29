import unittest
import random
import pandas as pd
import category_encoders as encoders

__author__ = 'willmcginnis'


class TestEncoders(unittest.TestCase):
    """
    """
    def create_dataset(self, n_rows=1000):
        """
        Creates a dataset with some categorical variables
        :return:
        """

        ds = [[
            random.random(),
            random.random(),
            random.choice(['A', 'B', 'C']),
            random.choice(['A', 'B', 'C']),
            random.choice(['A', 'B', 'C']),
            random.choice(['A', 'B', 'C'])
        ] for _ in range(n_rows)]

        df = pd.DataFrame(ds, columns=['A', 'B', 'C1', 'D', 'E', 'F'])
        return df

    def test_hashing(self):
        """
        Creates a dataset and encodes with with the hashing trick

        :return:
        """

        cols = ['C1', 'D', 'E', 'F']
        enc = encoders.HashingEncoder(verbose=1, n_components=128, cols=cols)
        X = self.create_dataset(n_rows=1000)

        X_test = enc.fit_transform(X, None)

        for dt in X_test.dtypes:
            numeric = False
            if dt == int or dt == float:
                numeric = True
            self.assertTrue(numeric)

    def test_ordinal(self):
        """

        :return:
        """

        cols = ['C1', 'D', 'E', 'F']
        enc = encoders.OrdinalEncoder(verbose=1, cols=cols)
        X = self.create_dataset(n_rows=1000)

        X_test = enc.fit_transform(X, None)

        for dt in X_test.dtypes:
            numeric = False
            if dt == int or dt == float:
                numeric = True
            self.assertTrue(numeric)

    def test_backward_difference(self):
        """

        :return:
        """

        cols = ['C1', 'D', 'E', 'F']
        enc = encoders.BackwardDifferenceEncoder(verbose=1, cols=cols)
        X = self.create_dataset(n_rows=1000)

        X_test = enc.fit_transform(X, None)

        for dt in X_test.dtypes:
            numeric = False
            if dt == int or dt == float:
                numeric = True
            self.assertTrue(numeric)

    def test_binary(self):
        """

        :return:
        """

        cols = ['C1', 'D', 'E', 'F']
        enc = encoders.BinaryEncoder(verbose=1, cols=cols)
        X = self.create_dataset(n_rows=1000)

        X_test = enc.fit_transform(X, None)

        for dt in X_test.dtypes:
            numeric = False
            if dt == int or dt == float:
                numeric = True
            self.assertTrue(numeric)

    def test_helmert(self):
        """

        :return:
        """

        cols = ['C1', 'D', 'E', 'F']
        enc = encoders.HelmertEncoder(verbose=1, cols=cols)
        X = self.create_dataset(n_rows=1000)

        X_test = enc.fit_transform(X, None)

        for dt in X_test.dtypes:
            numeric = False
            if dt == int or dt == float:
                numeric = True
            self.assertTrue(numeric)

    def test_polynomial(self):
        """

        :return:
        """

        cols = ['C1', 'D', 'E', 'F']
        enc = encoders.PolynomialEncoder(verbose=1, cols=cols)
        X = self.create_dataset(n_rows=1000)

        X_test = enc.fit_transform(X, None)

        for dt in X_test.dtypes:
            numeric = False
            if dt == int or dt == float:
                numeric = True
            self.assertTrue(numeric)

    def test_sum(self):
        """

        :return:
        """

        cols = ['C1', 'D', 'E', 'F']
        enc = encoders.SumEncoder(verbose=1, cols=cols)
        X = self.create_dataset(n_rows=1000)

        X_test = enc.fit_transform(X, None)

        for dt in X_test.dtypes:
            numeric = False
            if dt == int or dt == float:
                numeric = True
            self.assertTrue(numeric)