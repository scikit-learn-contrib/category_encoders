"""Helmert contrast coding"""


import pandas as pd
import numpy as np
from patsy.contrasts import Helmert
from category_encoders.ordinal import OrdinalEncoder
import category_encoders.utils as util

__author__ = 'willmcginnis'


class HelmertEncoder(util.BaseEncoder, util.UnsupervisedTransformerMixin):
    """Helmert contrast coding for encoding categorical features.

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
    >>> enc = HelmertEncoder(cols=['CHAS', 'RAD'], handle_unknown='value', handle_missing='value').fit(X, y)
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
    prefit_ordinal = True
    encoding_relation = util.EncodingRelation.ONE_TO_ONE

    def __init__(self, verbose=0, cols=None, mapping=None, drop_invariant=False, return_df=True,
                 handle_unknown='value', handle_missing='value'):
        super().__init__(verbose=verbose, cols=cols, drop_invariant=drop_invariant, return_df=return_df,
                         handle_unknown=handle_unknown, handle_missing=handle_missing)
        self.mapping = mapping
        self.ordinal_encoder = None

    def _fit(self, X, y=None, **kwargs):

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

            column_mapping = self.fit_helmert_coding(col, values, self.handle_missing, self.handle_unknown)
            mappings_out.append({'col': col, 'mapping': column_mapping, })

        self.mapping = mappings_out

    def _transform(self, X):
        X = self.ordinal_encoder.transform(X)

        if self.handle_unknown == 'error':
            if X[self.cols].isin([-1]).any().any():
                raise ValueError('Columns to be encoded can not contain new values')

        X = self.helmert_coding(X, mapping=self.mapping)
        return X

    @staticmethod
    def fit_helmert_coding(col, values, handle_missing, handle_unknown):
        if handle_missing == 'value':
            values = values[values > 0]

        values_to_encode = values.values

        if len(values) < 2:
            return pd.DataFrame(index=values_to_encode)

        if handle_unknown == 'indicator':
            values_to_encode = np.append(values_to_encode, -1)

        helmert_contrast_matrix = Helmert().code_without_intercept(values_to_encode)
        df = pd.DataFrame(data=helmert_contrast_matrix.matrix, index=values_to_encode,
                          columns=[f"{col}_{i}" for i in range(len(helmert_contrast_matrix.column_suffixes))])

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
    def helmert_coding(X_in, mapping):
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
