"""

.. module:: ordinal
  :synopsis:
  :platform:

"""

import pandas as pd
import copy
from sklearn.base import BaseEstimator, TransformerMixin
import random

__author__ = 'willmcginnis'


def ordinal_encoding(X_in, mapping=None, cols=None):
    """
    Ordinal encoding uses a single column of integers to represent the classes. An optional mapping dict can be passed
    in, in this case we use the knowledge that there is some true order to the classes themselves. Otherwise, the classes
    are assumed to have no true order and integers are selected at random.

    :param X:
    :return:
    """

    X = copy.deepcopy(X_in)

    if cols is None:
        cols = X.columns.values

    mapping_out = []
    if mapping is not None:
        for switch in mapping:
            for category in switch.get('mapping'):
                X.loc[X[switch.get('col')] == category[0], switch.get('col')] = str(category[1])
            X[switch.get('col')] = X[switch.get('col')].astype(int).reshape(-1, )
    else:
        for col in cols:
            categories = list(set(X[col].values))
            random.shuffle(categories)
            for idx, val in enumerate(categories):
                X.loc[X[col] == val, col] = str(idx)
            X[col] = X[col].astype(int).reshape(-1, )
            mapping_out.append({'col': col, 'mapping': [(x[1], x[0]) for x in list(enumerate(categories))]},)

    return X, mapping_out


class OrdinalEncoder(BaseEstimator, TransformerMixin):
    """
    Ordinal encoding uses a single column of integers to represent the classes. An optional mapping dict can be passed
    in, in this case we use the knowledge that there is some true order to the classes themselves. Otherwise, the classes
    are assumed to have no true order and integers are selected at random.
    """
    def __init__(self, verbose=0, mapping=None, cols=None):
        """

        :param verbose: foo
        :param mapping: bar
        :param cols: baz
        :return:
        """
        self.verbose = verbose
        self.cols = cols
        self.mapping = mapping

    def fit(self, X, y=None, **kwargs):
        """
        Fit doesn't actually do anything in this case.  So the same object is just returned as-is.

        :param X:
        :param y:
        :param kwargs:
        :return:
        """

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        _, categories = ordinal_encoding(X, mapping=self.mapping, cols=self.cols)
        self.mapping = categories

        return self

    def transform(self, X):
        """
        Will use the mapping (if available) and the column list (if available, otherwise every column) to encode the
        data ordinally.

        :param X:
        :return:
        """

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X, _ = ordinal_encoding(X, mapping=self.mapping, cols=self.cols)
        return X