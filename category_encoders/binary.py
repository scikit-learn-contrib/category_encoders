"""

.. module:: binary
  :synopsis:
  :platform:

"""

import copy
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

__author__ = 'willmcginnis'


def binary(X_in, cols=None):
    """
    Binary encoding encodes the integers as binary code with one column per digit.

    :param X:
    :return:
    """

    X = copy.deepcopy(X_in)

    if cols is None:
        cols = X.columns.values

    bin_cols = []
    for col in cols:
        # figure out how many digits we need to represent the classes present
        if X[col].max() == 0:
            digits = 1
        else:
            digits = int(np.ceil(np.log2(X[col].max())))

        # map the ordinal column into a list of these digits, of length digits
        X[col] = X[col].map(lambda x: list("{0:b}".format(int(x)))) \
            .map(lambda x: x if len(x) == digits else [0 for _ in range(digits - len(x))] + x)

        for dig in range(digits):
            X[col + '_%d' % (dig, )] = X[col].map(lambda x: int(x[dig]))
            bin_cols.append(col + '_%d' % (dig, ))

    X = X.reindex(columns=bin_cols)

    return X


class BinaryEncoder(BaseEstimator, TransformerMixin):
    """
    Binary encoding encodes the integers as binary code with one column per digit.

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

        return binary(X, cols=self.cols)