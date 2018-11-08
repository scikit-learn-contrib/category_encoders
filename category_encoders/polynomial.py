"""Polynomial contrast coding"""

import numpy as np
import pandas as pd
from patsy.contrasts import Poly
from category_encoders.ordinal import OrdinalEncoder
from sklearn.base import BaseEstimator, TransformerMixin
import category_encoders.utils as util

__author__ = 'willmcginnis'


class PolynomialEncoder(BaseEstimator, TransformerMixin):
    """Polynomial contrast coding for the encoding of categorical features.

    Parameters
    ----------

    verbose: int
        integer indicating verbosity of output. 0 for none.
    cols: list
        a list of columns to encode, if None, all string columns will be encoded.
    drop_invariant: bool
        boolean for whether or not to drop columns with 0 variance.
    return_df: bool
        boolean for whether to return a pandas DataFrame from transform (otherwise it will be a numpy array).
    handle_unknown: str
        options are 'error', 'ignore' and 'value', defaults to 'value'. Warning: if value is used,
        an extra column will be added in if the transform matrix has unknown categories.  This can cause
        unexpected changes in the dimension in some cases.
    handle_missing: str
        options are 'error', 'ignore', 'value', and 'indicator', defaults to 'indicator'. Warning: if indicator is used,
        an extra column will be added in if the transform matrix has unknown categories.  This can causes
        unexpected changes in dimension in some cases.

    Example
    -------
    >>> from category_encoders import *
    >>> import pandas as pd
    >>> from sklearn.datasets import load_boston
    >>> bunch = load_boston()
    >>> y = bunch.target
    >>> X = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    >>> enc = PolynomialEncoder(cols=['CHAS', 'RAD']).fit(X, y)
    >>> numeric_dataset = enc.transform(X)
    >>> print(numeric_dataset.info())
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 506 entries, 0 to 505
    Data columns (total 21 columns):
    intercept    506 non-null int64
    CRIM         506 non-null float64
    ZN           506 non-null float64
    INDUS        506 non-null float64
    CHAS_0       506 non-null float64
    NOX          506 non-null float64
    RM           506 non-null float64
    AGE          506 non-null float64
    DIS          506 non-null float64
    RAD_0        506 non-null float64
    RAD_1        506 non-null float64
    RAD_2        506 non-null float64
    RAD_3        506 non-null float64
    RAD_4        506 non-null float64
    RAD_5        506 non-null float64
    RAD_6        506 non-null float64
    RAD_7        506 non-null float64
    TAX          506 non-null float64
    PTRATIO      506 non-null float64
    B            506 non-null float64
    LSTAT        506 non-null float64
    dtypes: float64(20), int64(1)
    memory usage: 83.1 KB
    None

    References
    ----------

    .. [1] Contrast Coding Systems for categorical variables.  UCLA: Statistical Consulting Group. from
    https://stats.idre.ucla.edu/r/library/r-library-contrast-coding-systems-for-categorical-variables/.

    .. [2] Gregory Carey (2003). Coding Categorical Variables, from
    http://psych.colorado.edu/~carey/Courses/PSYC5741/handouts/Coding%20Categorical%20Variables%202006-03-03.pdf


    """
    def __init__(self, verbose=0, cols=None, mapping=None, drop_invariant=False, return_df=True,
                 handle_unknown='value', handle_missing='value'):
        self.return_df = return_df
        self.drop_invariant = drop_invariant
        self.drop_cols = []
        self.verbose = verbose
        self.mapping = mapping
        self.handle_unknown = handle_unknown
        self.handle_missing = handle_missing
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
        X = util.convert_input(X)

        self._dim = X.shape[1]

        # if columns aren't passed, just use every string column
        if self.cols is None:
            self.cols = util.get_obj_cols(X)
        else:
            self.cols = util.convert_cols_to_list(self.cols)

        if self.handle_missing == 'error':
            if X[self.cols].isnull().any().bool():
                raise ValueError('Columns to be encoded can not contain null')

        # train an ordinal pre-encoder
        self.ordinal_encoder = OrdinalEncoder(
            verbose=self.verbose,
            cols=self.cols,
            handle_unknown='value',
            handle_missing='value'
        )
        self.ordinal_encoder = self.ordinal_encoder.fit(X)

        ordinal_mapping = self.ordinal_encoder.category_mapping

        mappings_out = []
        for switch in ordinal_mapping:
            values = switch.get('mapping')
            column_mapping = self.fit_polynomial_coding(values, self.handle_missing, self.handle_unknown)
            mappings_out.append({'col': switch.get('col'), 'mapping': column_mapping, })

        self.mapping = mappings_out

        # drop all output columns with 0 variance.
        if self.drop_invariant:
            self.drop_cols = []
            X_temp = self.transform(X)
            generated_cols = util.get_generated_cols(X, X_temp, self.cols)
            self.drop_cols = [x for x in generated_cols if X_temp[x].var() <= 10e-5]

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

        if self.handle_missing == 'error':
            if X[self.cols].isnull().any().bool():
                raise ValueError('Columns to be encoded can not contain null')

        if self._dim is None:
            raise ValueError('Must train encoder before it can be used to transform data.')

        # first check the type
        X = util.convert_input(X)

        # then make sure that it is the right size
        if X.shape[1] != self._dim:
            raise ValueError('Unexpected input dimension %d, expected %d' % (X.shape[1], self._dim, ))

        if not self.cols:
            return X

        X = self.ordinal_encoder.transform(X)

        if self.handle_unknown == 'error':
            if X[self.cols].isin([-1]).any().any():
                raise ValueError('Columns to be encoded can not contain new values')

        X = self.polynomial_coding(X, self.mapping)

        if self.drop_invariant:
            for col in self.drop_cols:
                X.drop(col, 1, inplace=True)

        if self.return_df:
            return X
        else:
            return X.values

    @staticmethod
    def fit_polynomial_coding(values, handle_missing, handle_unknown):
        if handle_missing == 'value':
            del values[np.nan]

        if len(values) < 2:
            return pd.DataFrame()

        polynomial_contrast_matrix = Poly().code_without_intercept(values.get_values())
        df = pd.DataFrame(data=polynomial_contrast_matrix.matrix, columns=polynomial_contrast_matrix.column_suffixes)
        df.index += 1

        if handle_unknown == 'return_nan':
            df.loc[-1] = np.nan
        elif handle_unknown == 'value':
            df.loc[-1] = np.zeros(len(values) - 1)

        return df

    @staticmethod
    def polynomial_coding(X_in, mapping):
        """
        """

        X = X_in.copy(deep=True)

        cols = X.columns.values.tolist()

        X['intercept'] = pd.Series([1] * X.shape[0], index=X.index)

        for switch in mapping:
            col = switch.get('col')
            mod = switch.get('mapping')
            new_columns = []
            for i in range(len(mod.columns)):
                c = mod.columns[i]
                new_col = str(col) + '_%d' % (i, )
                X.loc[:, new_col] = mod[c].loc[X[col]].values
                new_columns.append(new_col)
            old_column_index = cols.index(col)
            cols[old_column_index: old_column_index + 1] = new_columns

        cols = ['intercept'] + cols
        X = X.reindex(columns=cols)

        return X
