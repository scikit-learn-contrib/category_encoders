"""Binary encoding"""

import copy
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from category_encoders.ordinal import OrdinalEncoder
from category_encoders.utils import get_obj_cols, convert_input

__author__ = 'willmcginnis'


class BinaryEncoder(BaseEstimator, TransformerMixin):
    """Binary encoding for categorical variables, similar to onehot, but stores categories as binary bitstrings.

    Parameters
    ----------

    verbose: int
        integer indicating verbosity of output. 0 for none.
    cols: list
        a list of columns to encode, if None, all string columns will be encoded
    drop_invariant: bool
        boolean for whether or not to drop columns with 0 variance
    return_df: bool
        boolean for whether to return a pandas DataFrame from transform (otherwise it will be a numpy array)

    Example
    -------
    >>>from category_encoders import *
    >>>import pandas as pd
    >>>from sklearn.datasets import load_boston
    >>>bunch = load_boston()
    >>>y = bunch.target
    >>>X = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    >>>enc = BinaryEncoder(cols=['CHAS', 'RAD']).fit(X, y)
    >>>numeric_dataset = enc.transform(X)
    >>>print(numeric_dataset.info())

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 506 entries, 0 to 505
    Data columns (total 16 columns):
    CHAS_0     506 non-null int64
    RAD_0      506 non-null int64
    RAD_1      506 non-null int64
    RAD_2      506 non-null int64
    RAD_3      506 non-null int64
    CRIM       506 non-null float64
    ZN         506 non-null float64
    INDUS      506 non-null float64
    NOX        506 non-null float64
    RM         506 non-null float64
    AGE        506 non-null float64
    DIS        506 non-null float64
    TAX        506 non-null float64
    PTRATIO    506 non-null float64
    B          506 non-null float64
    LSTAT      506 non-null float64
    dtypes: float64(11), int64(5)
    memory usage: 63.3 KB
    None

    """
    def __init__(self, verbose=0, cols=None, drop_invariant=False, return_df=True):
        self.return_df = return_df
        self.drop_invariant = drop_invariant
        self.drop_cols = []
        self.verbose = verbose
        self.cols = cols
        self.ordinal_encoder = None
        self._dim = None

    def fit(self, X, y=None, **kwargs):
        """Fit encoder according to X and y.

        Parameters
        ----------

        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------

        self : encoder
            Returns self.

        """

        # if the input dataset isn't already a dataframe, convert it to one (using default column names)
        # first check the type
        X = convert_input(X)

        self._dim = X.shape[1]

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
        X = convert_input(X)

        # then make sure that it is the right size
        if X.shape[1] != self._dim:
            raise ValueError('Unexpected input dimension %d, expected %d' % (X.shape[1], self._dim, ))

        if not self.cols:
            return X

        X = self.ordinal_encoder.transform(X)

        X = self.binary(X, cols=self.cols)

        if self.drop_invariant:
            for col in self.drop_cols:
                X.drop(col, 1, inplace=True)

        if self.return_df:
            return X
        else:
            return X.values

    def binary(self, X_in, cols=None):
        """
        Binary encoding encodes the integers as binary code with one column per digit.
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
            X[col] = X[col].map(lambda x: self.col_transform(x, digits))

            for dig in range(digits):
                X[str(col) + '_%d' % (dig, )] = X[col].map(lambda r: int(r[dig]) if r is not None else None)
                bin_cols.append(str(col) + '_%d' % (dig, ))

        X = X.reindex(columns=bin_cols + pass_thru)

        return X

    @staticmethod
    def col_transform(col, digits):
        """
        The lambda body to transform the column values
        """

        if col is None or float(col) < 0.0:
            return None
        else:

            col = list("{0:b}".format(int(col)))
            if len(col) == digits:
                return col
            else:
                return [0 for _ in range(digits - len(col))] + col