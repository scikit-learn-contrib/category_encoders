"""Sum contrast coding"""

import pandas as pd
import numpy as np
from patsy.contrasts import Sum
from sklearn.base import BaseEstimator, TransformerMixin
from category_encoders.ordinal import OrdinalEncoder
import category_encoders.utils as util

__author__ = 'willmcginnis'


class SumEncoder(BaseEstimator, TransformerMixin):
    """Sum contrast coding for the encoding of categorical features.

    Parameters
    ----------

    verbose: int
        integer indicating verbosity of the output. 0 for none.
    cols: list
        a list of columns to encode, if None, all string columns will be encoded.
    drop_invariant: bool
        boolean for whether or not to drop columns with 0 variance.
    return_df: bool
        boolean for whether to return a pandas DataFrame from transform (otherwise it will be a numpy array).
    handle_unknown: str
        options are 'error', 'return_nan', 'value', and 'indicator'. The default is 'value'. Warning: if indicator is used,
        an extra column will be added in if the transform matrix has unknown categories.  This can cause
        unexpected changes in dimension in some cases.
    handle_missing: str
        options are 'error', 'return_nan', 'value', and 'indicator'. The default is 'value'. Warning: if indicator is used,
        an extra column will be added in if the transform matrix has nan values.  This can cause
        unexpected changes in dimension in some cases.

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

    .. [1] Contrast Coding Systems for Categorical Variables, from
    https://stats.idre.ucla.edu/r/library/r-library-contrast-coding-systems-for-categorical-variables/

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
        self.feature_names = None

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
            if X[self.cols].isnull().any().any():
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
            col = switch.get('col')
            column_mapping = self.fit_sum_coding(col, values, self.handle_missing, self.handle_unknown)
            mappings_out.append({'col': switch.get('col'), 'mapping': column_mapping, })

        self.mapping = mappings_out

        X_temp = self.transform(X, override_return_df=True)
        self.feature_names = X_temp.columns.tolist()

        # drop all output columns with 0 variance.
        if self.drop_invariant:
            self.drop_cols = []
            generated_cols = util.get_generated_cols(X, X_temp, self.cols)
            self.drop_cols = [x for x in generated_cols if X_temp[x].var() <= 10e-5]
            try:
                [self.feature_names.remove(x) for x in self.drop_cols]
            except KeyError as e:
                if self.verbose > 0:
                    print("Could not remove column from feature names."
                    "Not found in generated cols.\n{}".format(e))

        return self

    def transform(self, X, override_return_df=False):
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
            if X[self.cols].isnull().any().any():
                raise ValueError('Columns to be encoded can not contain null')

        if self._dim is None:
            raise ValueError('Must train encoder before it can be used to transform data.')

        # first check the type
        X = util.convert_input(X)

        # then make sure that it is the right size
        if X.shape[1] != self._dim:
            raise ValueError('Unexpected input dimension %d, expected %d' % (X.shape[1], self._dim, ))

        if not list(self.cols):
            return X

        X = self.ordinal_encoder.transform(X)

        if self.handle_unknown == 'error':
            if X[self.cols].isin([-1]).any().any():
                raise ValueError('Columns to be encoded can not contain new values')

        X = self.sum_coding(X, mapping=self.mapping)

        if self.drop_invariant:
            for col in self.drop_cols:
                X.drop(col, 1, inplace=True)

        if self.return_df or override_return_df:
            return X
        else:
            return X.values

    @staticmethod
    def fit_sum_coding(col, values, handle_missing, handle_unknown):
        if handle_missing == 'value':
            values = values[values > 0]

        values_to_encode = values.values

        if len(values) < 2:
            return pd.DataFrame(index=values_to_encode)

        if handle_unknown == 'indicator':
            values_to_encode = np.append(values_to_encode, -1)

        sum_contrast_matrix = Sum().code_without_intercept(values_to_encode.tolist())
        df = pd.DataFrame(data=sum_contrast_matrix.matrix, index=values_to_encode,
                          columns=[str(col) + '_%d' % (i, ) for i in range(len(sum_contrast_matrix.column_suffixes))])

        if handle_unknown == 'return_nan':
            df.loc[-1] = np.nan
        elif handle_unknown == 'value':
            df.loc[-1] = np.zeros(len(values_to_encode) - 1)

        if handle_missing == 'return_nan':
            df.loc[values.loc[np.nan]] = np.nan
        elif handle_missing == 'value':
            df.loc[-2] = np.zeros(len(values_to_encode) - 1)

        return df

    @staticmethod
    def sum_coding(X_in, mapping):
        """
        """

        X = X_in.copy(deep=True)

        cols = X.columns.values.tolist()

        X['intercept'] = pd.Series([1] * X.shape[0], index=X.index)

        for switch in mapping:
            col = switch.get('col')
            mod = switch.get('mapping')

            base_df = mod.reindex(X[col])
            base_df.set_index(X.index, inplace=True)
            X = pd.concat([base_df, X], axis=1)

            old_column_index = cols.index(col)
            cols[old_column_index: old_column_index + 1] = mod.columns

        cols = ['intercept'] + cols

        return X.reindex(columns=cols)

    def get_feature_names(self):
        """
        Returns the names of all transformed / added columns.

        Returns
        -------
        feature_names: list
            A list with all feature names transformed or added.
            Note: potentially dropped features are not included!

        """
        if not isinstance(self.feature_names, list):
            raise ValueError("Estimator has to be fitted to return feature names.")
        else:
            return self.feature_names
