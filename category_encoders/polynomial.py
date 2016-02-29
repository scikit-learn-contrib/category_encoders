"""

.. module:: polynomial
  :synopsis:
  :platform:

"""

import copy
from sklearn.base import BaseEstimator, TransformerMixin
from patsy.highlevel import dmatrix

__author__ = 'willmcginnis'


def polynomial_coding(X_in, cols=None):
    """

    :param X:
    :return:
    """

    X = copy.deepcopy(X_in)
    if cols is None:
        cols = X.columns.values

    bin_cols = []
    for col in cols:
        mod = dmatrix("C(%s, Poly)" % (col, ), X)
        for dig in range(len(mod[0])):
            X[col + '_%d' % (dig, )] = mod[:, dig]
            bin_cols.append(col + '_%d' % (dig, ))

    X = X.reindex(columns=bin_cols)

    return X


class PolynomialEncoder(BaseEstimator, TransformerMixin):
    """

    """
    def __init__(self, verbose=0, cols=None):
        """

        :param verbose:
        :param cols:
        :return:
        """

        self.verbose = verbose
        self.cols = cols

    def fit(self, X, y=None, **kwargs):
        """

        :param X:
        :param y:
        :param kwargs:
        :return:
        """

        return self

    def transform(self, X):
        """

        :param X:
        :return:
        """

        return polynomial_coding(X, cols=self.cols)
