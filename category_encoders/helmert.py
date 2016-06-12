"""

.. module:: helmert
  :synopsis:
  :platform:

"""

import copy
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from patsy.highlevel import dmatrix
from category_encoders.ordinal import OrdinalEncoder
from category_encoders.utils import get_obj_cols

__author__ = 'willmcginnis'


def helmert_coding(X_in, cols=None):
    """

    :param X:
    :return:
    """

    X = copy.deepcopy(X_in)

    if cols is None:
        cols = X.columns.values
        pass_thru = []
    else:
        pass_thru = [col for col in X.columns.values if col not in cols]

    bin_cols = []
    for col in cols:
        mod = dmatrix("C(%s, Helmert)" % (col, ), X)
        for dig in range(len(mod[0])):
            X[col + '_%d' % (dig, )] = mod[:, dig]
            bin_cols.append(col + '_%d' % (dig, ))

    X = X.reindex(columns=bin_cols + pass_thru)

    return X


class HelmertEncoder(BaseEstimator, TransformerMixin):
    """

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
        self.ordinal_encoder = OrdinalEncoder(verbose=verbose, cols=cols)

    def fit(self, X, y=None, **kwargs):
        """

        :param X:
        :param y:
        :param kwargs:
        :return:
        """

        self.ordinal_encoder = self.ordinal_encoder.fit(X)

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

        X = helmert_coding(X, cols=self.cols)

        if self.drop_invariant:
            for col in self.drop_cols:
                X.drop(col, 1, inplace=True)

        if self.return_df:
            return X
        else:
            return X.values