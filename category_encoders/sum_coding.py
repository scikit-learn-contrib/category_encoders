"""Sum contrast coding"""

import pandas as pd
import numpy as np
from patsy.contrasts import Sum
from sklearn.base import BaseEstimator, TransformerMixin
from category_encoders.ordinal import OrdinalEncoder
from category_encoders.utils import get_obj_cols, convert_input, get_generated_cols

__author__ = 'willmcginnis'


class SumEncoder(BaseEstimator, TransformerMixin):
    """Sum contrast coding for the encoding of categorical features.

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
    impute_missing: bool
        boolean for whether or not to apply the logic for handle_unknown, will be deprecated in the future.
    handle_unknown: str
        options are 'error', 'ignore' and 'impute', defaults to 'impute', which will impute the category -1. Warning: if
        impute is used, an extra column will be added in if the transform matrix has unknown categories.  This can cause
        unexpected changes in the dimension in some cases.

    Example
    -------
    >>> from category_encoders import *
    >>> import pandas as pd
    >>> from sklearn.datasets import load_boston
    >>> bunch = load_boston()
    >>> y = bunch.target
    >>> X = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    >>> enc = SumEncoder(cols=['CHAS', 'RAD']).fit(X, y)
    >>> numeric_dataset = enc.transform(X)
    >>> print(numeric_dataset.info())
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
    https://stats.idre.ucla.edu/r/library/r-library-contrast-coding-systems-for-categorical-variables/.

    .. [2] Gregory Carey (2003). Coding Categorical Variables, from
    http://psych.colorado.edu/~carey/Courses/PSYC5741/handouts/Coding%20Categorical%20Variables%202006-03-03.pdf


    """
    def __init__(self, verbose=0, cols=None, mapping=None, drop_invariant=False, return_df=True, impute_missing=True, handle_unknown='impute'):
        self.return_df = return_df
        self.drop_invariant = drop_invariant
        self.drop_cols = []
        self.verbose = verbose
        self.mapping = mapping
        self.impute_missing = impute_missing
        self.handle_unknown = handle_unknown
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
        self.ordinal_encoder = OrdinalEncoder(
            verbose=self.verbose,
            cols=self.cols,
            impute_missing=self.impute_missing,
            handle_unknown=self.handle_unknown
        )
        self.ordinal_encoder = self.ordinal_encoder.fit(X)

        ordinal_mapping = self.ordinal_encoder.category_mapping

        mappings_out = []
        for switch in ordinal_mapping:
            values = [x[1] for x in switch.get('mapping')]
            column_mapping = self.fit_sum_coding(values)
            mappings_out.append({'col': switch.get('col'), 'mapping': column_mapping, })

        self.mapping = mappings_out

        # drop all output columns with 0 variance.
        if self.drop_invariant:
            self.drop_cols = []
            X_temp = self.transform(X)
            generated_cols = get_generated_cols(X, X_temp, self.cols)
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

        X = self.sum_coding(X, mapping=self.mapping)

        if self.drop_invariant:
            for col in self.drop_cols:
                X.drop(col, 1, inplace=True)

        if self.return_df:
            return X
        else:
            return X.values

    @staticmethod
    def fit_sum_coding(values):
        if len(values) < 2:
            return pd.DataFrame()

        sum_contrast_matrix = Sum().code_without_intercept(values)
        df = pd.DataFrame(data=sum_contrast_matrix.matrix, columns=sum_contrast_matrix.column_suffixes)
        df.index += 1
        df.loc[0] = np.zeros(len(values) - 1)
        return df

    @staticmethod
    def sum_coding(X_in, mapping):
        """
        """

        X = X_in.copy(deep=True)

        cols = X.columns.values.tolist()

        X['intercept'] = pd.Series([1] * X.shape[0])

        for switch in mapping:
            col = switch.get('col')
            mod = switch.get('mapping')
            new_columns = []
            for i in range(len(mod.columns)):
                c = mod.columns[i]
                new_col = str(col) + '_%d' % (i, )
                X[new_col] = mod[c].loc[X[col]].values
                new_columns.append(new_col)
            old_column_index = cols.index(col)
            cols[old_column_index: old_column_index + 1] = new_columns

        cols = ['intercept'] + cols
        X = X.reindex(columns=cols)

        return X
