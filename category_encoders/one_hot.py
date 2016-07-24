"""

.. module:: one_hot
  :synopsis:
  :platform:

"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from category_encoders.ordinal import OrdinalEncoder
from category_encoders.utils import get_obj_cols

__author__ = 'willmcginnis'


class OneHotEncoder(BaseEstimator, TransformerMixin):
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
        self._dim = None

    def fit(self, X, y=None, **kwargs):
        """

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

        self.ordinal_encoder = OrdinalEncoder(verbose=self.verbose, cols=self.cols)
        self.ordinal_encoder = self.ordinal_encoder.fit(X)

        if self.drop_invariant:
            self.drop_cols = []
            X_temp = self.transform(X)
            self.drop_cols = [x for x in X_temp.columns.values if X_temp[x].var() <= 10e-5]

        return self

    def transform(self, X):
        """Perform the transformation to new categorical data.

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

        X = self.ordinal_encoder.transform(X)

        X = self.get_dummies(X, cols=self.cols)

        if self.drop_invariant:
            for col in self.drop_cols:
                X.drop(col, 1, inplace=True)

        if self.return_df:
            return X
        else:
            return X.values

    @staticmethod
    def get_dummies(X_in, cols=None):
        """

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
            classes = set(X[col].values.tolist())
            for class_ in classes:
                n_col_name = col + '_%s' % (class_, )
                X[n_col_name] = 0
                X.loc[X[col] == class_, n_col_name] = 1
                bin_cols.append(n_col_name)

        X = X.reindex(columns=bin_cols + pass_thru)

        return X