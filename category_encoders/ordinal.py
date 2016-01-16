import copy
from sklearn.base import BaseEstimator, TransformerMixin
import random

__author__ = 'willmcginnis'


def ordinal_encoding(X_in, mapping=None):
    """
    Ordinal encoding uses a single column of integers to represent the classes. An optional mapping dict can be passed
    in, in this case we use the knowledge that there is some true order to the classes themselves. Otherwise, the classes
    are assumed to have no true order and integers are selected at random.

    :param X:
    :return:
    """

    X = copy.deepcopy(X_in)

    if mapping is not None:
        for switch in mapping:
            for category in switch.get('mapping'):
                X.loc[X[switch.get('col')] == category[0], switch.get('col')] = str(category[1])
            X[switch.get('col')] = X[switch.get('col')].astype(int).reshape(-1, )
    else:
        for col in X.columns.values:
            categories = list(set(X[col].values))
            random.shuffle(categories)
            for idx, val in enumerate(categories):
                X.loc[X[col] == val, col] = str(idx)
            X[col] = X[col].astype(int).reshape(-1, )

    return X


class OrdinalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, verbose=0):
        self.verbose = verbose

    def fit(self, X, y=None, **kwargs):
        return self

    def transform(self, X):
        return ordinal_encoding(X)