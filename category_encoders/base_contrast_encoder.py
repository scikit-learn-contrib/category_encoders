"""Base encoder for various contrast coding schemes"""
from abc import abstractmethod

import pandas as pd
from patsy.contrasts import ContrastMatrix
import numpy as np
from category_encoders.ordinal import OrdinalEncoder
import category_encoders.utils as util
import warnings

__author__ = 'paulwestenthanner'


class BaseContrastEncoder(util.BaseEncoder, util.UnsupervisedTransformerMixin):
    """Base class for various contrast encoders

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

    References
    ----------

    .. [1] Contrast Coding Systems for Categorical Variables, from
    https://stats.idre.ucla.edu/r/library/r-library-contrast-coding-systems-for-categorical-variables/

    .. [2] Gregory Carey (2003). Coding Categorical Variables, from
    http://ibgwww.colorado.edu/~carey/p5741ndir/Coding_Categorical_Variables.pdf

    """
    prefit_ordinal = True
    encoding_relation = util.EncodingRelation.ONE_TO_N_UNIQUE

    def __init__(self, verbose=0, cols=None, mapping=None, drop_invariant=False, return_df=True,
                 handle_unknown='value', handle_missing='value'):
        super().__init__(verbose=verbose, cols=cols, drop_invariant=drop_invariant, return_df=return_df,
                         handle_unknown=handle_unknown, handle_missing=handle_missing)
        self.mapping = mapping
        self.ordinal_encoder = None

    def _fit(self, X, y=None, **kwargs):
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

            column_mapping = self.fit_contrast_coding(col, values, self.handle_missing, self.handle_unknown)
            mappings_out.append({'col': col, 'mapping': column_mapping, })

        self.mapping = mappings_out

    def _transform(self, X) -> pd.DataFrame:
        X = self.ordinal_encoder.transform(X)
        if self.handle_unknown == 'error':
            if X[self.cols].isin([-1]).any().any():
                raise ValueError('Columns to be encoded can not contain new values')

        X = self.transform_contrast_coding(X, mapping=self.mapping)
        return X

    @abstractmethod
    def get_contrast_matrix(self, values_to_encode: np.ndarray) -> ContrastMatrix:
        raise NotImplementedError

    def fit_contrast_coding(self, col, values, handle_missing, handle_unknown):
        if handle_missing == 'value':
            values = values[values > 0]

        values_to_encode = values.to_numpy()

        if len(values) < 2:
            return pd.DataFrame(index=values_to_encode)

        if handle_unknown == 'indicator':
            values_to_encode = np.append(values_to_encode, -1)

        contrast_matrix = self.get_contrast_matrix(values_to_encode)
        df = pd.DataFrame(data=contrast_matrix.matrix, index=values_to_encode,
                          columns=[f"{col}_{i}" for i in range(len(contrast_matrix.column_suffixes))])

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
    def transform_contrast_coding(X, mapping):
        cols = X.columns.tolist()

        # See issue 370 if it is necessary to add an intercept or not.
        X['intercept'] = pd.Series([1] * X.shape[0], index=X.index)
        warnings.warn("Intercept column might not be added anymore in future releases (c.f. issue #370)",
                      category=FutureWarning)

        for switch in mapping:
            col = switch.get('col')
            mod = switch.get('mapping')

            # reindex actually applies the mapping
            base_df = mod.reindex(X[col])
            base_df = base_df.set_index(X.index)
            X = pd.concat([base_df, X], axis=1)

            old_column_index = cols.index(col)
            cols[old_column_index: old_column_index + 1] = mod.columns

        # this could lead to problems if an intercept column is already present
        # (e.g. if another column has been encoded with another contrast coding scheme)
        cols = ['intercept'] + cols

        return X.reindex(columns=cols)
