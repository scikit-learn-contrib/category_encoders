import copy
from sklearn.base import BaseEstimator, TransformerMixin
from patsy.highlevel import dmatrix

__author__ = 'willmcginnis'


def backward_difference_coding(X_in, cols=None):
    """

    :param X:
    :return:
    """

    X = copy.deepcopy(X_in)

    if cols is None:
        cols = X.columns.values

    bin_cols = []
    for col in cols:
        mod = dmatrix("C(%s, Diff)" % (col, ), X)
        for dig in range(len(mod[0])):
            X[col + '_%d' % (dig, )] = mod[:, dig]
            bin_cols.append(col + '_%d' % (dig, ))

    X = X.reindex(columns=bin_cols)
    X.fillna(0.0)
    return X


class BackwardDifferenceEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, verbose=0, cols=None):
        self.verbose = verbose
        self.cols = cols

    def fit(self, X, y=None, **kwargs):
        return self

    def transform(self, X):
        return backward_difference_coding(X, cols=self.cols)