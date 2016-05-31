"""

.. module:: hashing
  :synopsis:
  :platform:

"""

import copy
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from category_encoders.utils import get_obj_cols

__author__ = 'willmcginnis'


def hashing_trick_4(X_in):
    return hashing_trick(X_in, N=4)


def hashing_trick_8(X_in):
    return hashing_trick(X_in, N=8)


def hashing_trick_16(X_in):
    return hashing_trick(X_in, N=16)


def hashing_trick_32(X_in):
    return hashing_trick(X_in, N=32)


def hashing_trick_64(X_in):
    return hashing_trick(X_in, N=64)


def hashing_trick_128(X_in):
    return hashing_trick(X_in, N=128)


def hashing_trick(X_in, N=2, cols=None, make_copy=False):
    """
    A basic hashing implementation with configurable dimensionality/precision

    :param X_in:
    :return:
    """

    if make_copy:
        X = copy.deepcopy(X_in)
    else:
        X = X_in

    if cols is None:
        cols = X.columns.values

    def hash_fn(x):
        tmp = [0 for _ in range(N)]
        for val in x.values:
            tmp[hash(val) % N] += 1
        return pd.Series(tmp, index=new_cols)

    new_cols = ['col_%d' % d for d in range(N)]

    X_cat = X.reindex(columns=cols)
    X_num = X.reindex(columns=[x for x in X.columns.values if x not in cols])

    X_cat = X_cat.apply(hash_fn, axis=1)
    X_cat.columns = new_cols

    X = pd.merge(X_cat, X_num, left_index=True, right_index=True)
    return X


class HashingEncoder(BaseEstimator, TransformerMixin):
    """
    A basic hashing implementation with configurable dimensionality/precision

    """
    def __init__(self, verbose=0, n_components=8, cols=None, drop_invariant=False):
        """

        :param verbose:
        :param n_components:
        :param cols:
        :return:
        """

        self.drop_invariant = drop_invariant
        self.drop_cols = []
        self.verbose = verbose
        self.n_components = n_components
        self.cols = cols

    def fit(self, X, y=None, **kwargs):
        """

        :param X:
        :param y:
        :param kwargs:
        :return:
        """

        # if the input dataset isn't already a dataframe, convert it to one (using default column names)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # if columns aren't passed, just use every string column
        if self.cols is None:
            self.cols = get_obj_cols(X)

        # drop all output columns with 0 variance.
        if self.drop_invariant:
            self.drop_cols = []
            X_temp = self.transform(X)
            self.drop_cols = [x for x in X_temp.columns.values if X_temp[x].var() <= 10e-5]

        return self

    def transform(self, X):
        """

        :param X:
        :return:
        """

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if self.cols == []:
            return X

        X = hashing_trick(X, N=self.n_components, cols=self.cols)

        if self.drop_invariant:
            for col in self.drop_cols:
                X.drop(col, 1, inplace=True)

        return X