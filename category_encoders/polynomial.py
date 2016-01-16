import copy
from sklearn.base import BaseEstimator, TransformerMixin
from patsy.highlevel import dmatrix

__author__ = 'willmcginnis'


def polynomial_coding(X_in):
    """

    :param X:
    :return:
    """

    X = copy.deepcopy(X_in)

    bin_cols = []
    for col in X.columns.values:
        mod = dmatrix("C(%s, Poly)" % (col, ), X)
        for dig in range(len(mod[0])):
            X[col + '_%d' % (dig, )] = mod[:, dig]
            bin_cols.append(col + '_%d' % (dig, ))

    X = X.reindex(columns=bin_cols)

    return X


class PolynomialEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, verbose=0):
        self.verbose = verbose

    def fit(self, X, y=None, **kwargs):
        return self

    def transform(self, X):
        return polynomial_coding(X)