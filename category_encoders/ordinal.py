"""

.. module:: ordinal
  :synopsis:
  :platform:

"""

import pandas as pd
import copy
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import random
from category_encoders.utils import get_obj_cols

__author__ = 'willmcginnis'


class OrdinalEncoder(BaseEstimator, TransformerMixin):
    """
    Ordinal encoding uses a single column of integers to represent the classes. An optional mapping dict can be passed
    in, in this case we use the knowledge that there is some true order to the classes themselves. Otherwise, the classes
    are assumed to have no true order and integers are selected at random.
    """
    def __init__(self, verbose=0, mapping=None, cols=None, drop_invariant=False, return_df=True, impute_missing=True):
        """

        :param verbose: (optional, default=0) integer indicating verbosity of output. 0 for none.
        :param cols: (optional, default=None) a list of columns to encode, if None, all string columns will be encoded
        :param drop_invariant: (optional, default=False) boolean for whether or not to drop columns with 0 variance
        :param return_df: (optional, default=True) boolean for whether to return a pandas DataFrame from transform (otherwise it will be a numpy array)
        :param impute_missing: (optional, default=True) will impute missing values with the category -1
        :return:
        """

        self.return_df = return_df
        self.drop_invariant = drop_invariant
        self.drop_cols = []
        self.verbose = verbose
        self.cols = cols
        self.mapping = mapping
        self.impute_missing = impute_missing
        self._dim = None

    def fit(self, X, y=None, **kwargs):
        """
        Fit doesn't actually do anything in this case.  So the same object is just returned as-is.

        :param X:
        :param y:
        :param kwargs:
        :return:
        """

        # first check the type
        if not isinstance(X, pd.DataFrame):
            if isinstance(X, list):
                X = pd.DataFrame(np.array(X))
            elif isinstance(X, (np.generic, np.ndarray)):
                X = pd.DataFrame(X)
            else:
                raise ValueError('Unexpected input type: %s' % (str(type(X))))

        self._dim = X.shape[1]

        # if columns aren't passed, just use every string column
        if self.cols is None:
            self.cols = get_obj_cols(X)

        _, categories = self.ordinal_encoding(X, mapping=self.mapping, cols=self.cols, impute_missing=self.impute_missing)
        self.mapping = categories

        # drop all output columns with 0 variance.
        if self.drop_invariant:
            self.drop_cols = []
            X_temp = self.transform(X)
            self.drop_cols = [x for x in X_temp.columns.values if X_temp[x].var() <= 10e-5]

        return self

    def transform(self, X):
        """Perform the transformation to new categorical data.

        Will use the mapping (if available) and the column list (if available, otherwise every column) to encode the
        data ordinally.

        Parameters
        ----------

        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        p : array, shape = [n_samples, n_numeric + N]
            Transformed values with encoding applied.

        """


        if self._dim is None:
            raise ValueError('Must train encoder before it can be used to transform data.')

        # first check the type
        if not isinstance(X, pd.DataFrame):
            if isinstance(X, list):
                X = pd.DataFrame(np.array(X))
            elif isinstance(X, (np.generic, np.ndarray)):
                X = pd.DataFrame(X)
            else:
                raise ValueError('Unexpected input type: %s' % (str(type(X))))

        # then make sure that it is the right size
        if X.shape[1] != self._dim:
            raise ValueError('Unexpected input dimension %d, expected %d' % (X.shape[1], self._dim, ))

        if not self.cols:
            return X

        X, _ = self.ordinal_encoding(X, mapping=self.mapping, cols=self.cols)

        if self.drop_invariant:
            for col in self.drop_cols:
                X.drop(col, 1, inplace=True)

        if self.return_df:
            return X
        else:
            return X.values

    @staticmethod
    def ordinal_encoding(X_in, mapping=None, cols=None, impute_missing=True):
        """
        Ordinal encoding uses a single column of integers to represent the classes. An optional mapping dict can be passed
        in, in this case we use the knowledge that there is some true order to the classes themselves. Otherwise, the classes
        are assumed to have no true order and integers are selected at random.

        :param X:
        :return:
        """

        X = X_in.copy(deep=True)

        if cols is None:
            cols = X.columns.values

        mapping_out = []
        if mapping is not None:
            for switch in mapping:
                for category in switch.get('mapping'):
                    X.loc[X[switch.get('col')] == category[0], switch.get('col')] = str(category[1])
                if impute_missing:
                    X[switch.get('col')].fillna(-1, inplace=True)
                X[switch.get('col')] = X[switch.get('col')].astype(int).reshape(-1, )
        else:
            for col in cols:
                categories = list(set(X[col].values))
                random.shuffle(categories)
                for idx, val in enumerate(categories):
                    X.loc[X[col] == val, col] = str(idx)

                if impute_missing:
                    X[col].fillna(-1, inplace=True)

                X[col] = X[col].astype(int).reshape(-1, )

                mapping_out.append({'col': col, 'mapping': [(x[1], x[0]) for x in list(enumerate(categories))]},)

        return X, mapping_out