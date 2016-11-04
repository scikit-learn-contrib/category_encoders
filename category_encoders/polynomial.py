"""Polynomial contrast coding"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from patsy.highlevel import dmatrix
from category_encoders.utils import get_obj_cols, convert_input

__author__ = 'willmcginnis'


class PolynomialEncoder(BaseEstimator, TransformerMixin):
    """Polynomial contrast coding for the encoding of categorical features

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
    >>>enc = PolynomialEncoder(cols=['CHAS', 'RAD']).fit(X, y)
    >>>numeric_dataset = enc.transform(X)
    >>>print(numeric_dataset.info())

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 506 entries, 0 to 505
    Data columns (total 22 columns):
    col_CHAS_0     506 non-null float64
    col_CHAS_1     506 non-null float64
    col_RAD_0      506 non-null float64
    col_RAD_1      506 non-null float64
    col_RAD_2      506 non-null float64
    col_RAD_3      506 non-null float64
    col_RAD_4      506 non-null float64
    col_RAD_5      506 non-null float64
    col_RAD_6      506 non-null float64
    col_RAD_7      506 non-null float64
    col_RAD_8      506 non-null float64
    col_CRIM       506 non-null float64
    col_ZN         506 non-null float64
    col_INDUS      506 non-null float64
    col_NOX        506 non-null float64
    col_RM         506 non-null float64
    col_AGE        506 non-null float64
    col_DIS        506 non-null float64
    col_TAX        506 non-null float64
    col_PTRATIO    506 non-null float64
    col_B          506 non-null float64
    col_LSTAT      506 non-null float64
    dtypes: float64(22)
    memory usage: 87.0 KB
    None

    References
    ----------

    .. [1] Contrast Coding Systems for categorical variables.  UCLA: Statistical Consulting Group. from
    http://www.ats.ucla.edu/stat/r/library/contrast_coding.

    .. [2] Gregory Carey (2003). Coding Categorical Variables, from
    http://psych.colorado.edu/~carey/Courses/PSYC5741/handouts/Coding%20Categorical%20Variables%202006-03-03.pdf


    """
    def __init__(self, verbose=0, cols=None, drop_invariant=False, return_df=True):
        self.return_df = return_df
        self.drop_invariant = drop_invariant
        self.drop_cols = []
        self.verbose = verbose
        self.cols = cols
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

        X = self.polynomial_coding(X, cols=self.cols)

        if self.drop_invariant:
            for col in self.drop_cols:
                X.drop(col, 1, inplace=True)

        if self.return_df:
            return X
        else:
            return X.values

    @staticmethod
    def polynomial_coding(X_in, cols=None):
        """
        """

        X = X_in.copy(deep=True)

        X.columns = ['col_' + str(x) for x in X.columns.values]
        cols = ['col_' + str(x) for x in cols]

        if cols is None:
            cols = X.columns.values
            pass_thru = []
        else:
            pass_thru = [col for col in X.columns.values if col not in cols]

        X.fillna(-1, inplace=True)

        bin_cols = []
        for col in cols:
            mod = dmatrix("C(%s, Poly)" % (col, ), X)
            for dig in range(len(mod[0])):
                X[str(col) + '_%d' % (dig, )] = mod[:, dig]
                bin_cols.append(str(col) + '_%d' % (dig, ))

        X = X.reindex(columns=bin_cols + pass_thru)

        return X
