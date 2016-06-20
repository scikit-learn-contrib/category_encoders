"""

.. module:: binary
  :synopsis:
  :platform:

"""

import copy
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from category_encoders.ordinal import OrdinalEncoder
from category_encoders.utils import get_obj_cols

__author__ = 'willmcginnis'


def binary(X_in, cols=None):
    """
    Binary encoding encodes the integers as binary code with one column per digit.

    :param X:
    :return:
    """

    X = X_in.copy(deep=True)

    if cols is None:
        cols = X.columns.values
        pass_thru = []
    else:
        pass_thru = [col for col in X.columns.values if col not in cols]

    bin_cols = []
    for col in cols:
        # figure out how many digits we need to represent the classes present
        digits = int(np.ceil(np.log2(len(X[col].unique()))))

        # map the ordinal column into a list of these digits, of length digits
        X[col] = X[col].map(lambda x: list("{0:b}".format(int(x)))) \
            .map(lambda x: x if len(x) == digits else [0 for _ in range(digits - len(x))] + x)

        for dig in range(digits):
            X[col + '_%d' % (dig, )] = X[col].map(lambda x: int(x[dig]))
            bin_cols.append(col + '_%d' % (dig, ))

    X = X.reindex(columns=bin_cols + pass_thru)

    return X


class BinaryEncoder(BaseEstimator, TransformerMixin):
    """
    Binary encoding encodes the integers as binary code with one column per digit.

    """
    def __init__(self, verbose=0, cols=None, drop_invariant=False, return_df=True):
        """

        :param verbose: (optional, default=0) integer indicating verbosity of output. 0 for none.
        :param cols: (optional, default=None) a list of columns to encode, if None, all string columns will be encoded
        :param drop_invariant: (optional, default=False) boolean for whether or not to drop columns with 0 variance
        :param return_df: (optional, default=True) boolean for whether to return a pandas DataFrame from transform (otherwise it will be a numpy array)
        :return:
        """

        self.return_df = return_df
        self.drop_invariant = drop_invariant
        self.drop_cols = []
        self.verbose = verbose
        self.cols = cols
        self.ordinal_encoder = None

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

        # train an ordinal pre-encoder
        self.ordinal_encoder = OrdinalEncoder(verbose=self.verbose, cols=self.cols)
        self.ordinal_encoder = self.ordinal_encoder.fit(X)

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

        if not self.cols:
            return X

        X = self.ordinal_encoder.transform(X)

        X = binary(X, cols=self.cols)

        if self.drop_invariant:
            for col in self.drop_cols:
                X.drop(col, 1, inplace=True)

        if self.return_df:
            return X
        else:
            return X.values