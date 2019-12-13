"""BaseX encoding"""

import pandas as pd
import numpy as np
import math
import re
from sklearn.base import BaseEstimator, TransformerMixin
from category_encoders.ordinal import OrdinalEncoder
import category_encoders.utils as util
import warnings

__author__ = 'willmcginnis'


class BaseNEncoder(BaseEstimator, TransformerMixin):
    """Base-N encoder encodes the categories into arrays of their base-N representation.  A base of 1 is equivalent to
    one-hot encoding (not really base-1, but useful), a base of 2 is equivalent to binary encoding. N=number of actual
    categories is equivalent to vanilla ordinal encoding.

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
    base: int
        when the downstream model copes well with nonlinearities (like decision tree), use higher base.
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
    >>> enc = BaseNEncoder(cols=['CHAS', 'RAD']).fit(X, y)
    >>> numeric_dataset = enc.transform(X)
    >>> print(numeric_dataset.info())
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 506 entries, 0 to 505
    Data columns (total 18 columns):
    CRIM       506 non-null float64
    ZN         506 non-null float64
    INDUS      506 non-null float64
    CHAS_0     506 non-null int64
    CHAS_1     506 non-null int64
    NOX        506 non-null float64
    RM         506 non-null float64
    AGE        506 non-null float64
    DIS        506 non-null float64
    RAD_0      506 non-null int64
    RAD_1      506 non-null int64
    RAD_2      506 non-null int64
    RAD_3      506 non-null int64
    RAD_4      506 non-null int64
    TAX        506 non-null float64
    PTRATIO    506 non-null float64
    B          506 non-null float64
    LSTAT      506 non-null float64
    dtypes: float64(11), int64(7)
    memory usage: 71.3 KB
    None

    """

    def __init__(self, verbose=0, cols=None, mapping=None, drop_invariant=False, return_df=True, base=2,
                 handle_unknown='value', handle_missing='value'):
        self.return_df = return_df
        self.drop_invariant = drop_invariant
        self.drop_cols = []
        self.verbose = verbose
        self.handle_unknown = handle_unknown
        self.handle_missing = handle_missing
        self.cols = cols
        self.mapping = mapping
        self.ordinal_encoder = None
        self._dim = None
        self.base = base
        self._encoded_columns = None
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

        self.mapping = self.fit_base_n_encoding(X)

        # do a transform on the training data to get a column list
        X_temp = self.transform(X, override_return_df=True)
        self._encoded_columns = X_temp.columns.values
        self.feature_names = list(X_temp.columns)

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

    def fit_base_n_encoding(self, X):
        mappings_out = []

        for switch in self.ordinal_encoder.category_mapping:
            col = switch.get('col')
            values = switch.get('mapping')

            if self.handle_missing == 'value':
                values = values[values > 0]

            if self.handle_unknown == 'indicator':
                values = np.append(values, -1)

            digits = self.calc_required_digits(values)
            X_unique = pd.DataFrame(index=values,
                                    columns=[str(col) + '_%d' % x for x in range(digits)],
                                    data=np.array([self.col_transform(x, digits) for x in range(1, len(values) + 1)]))

            if self.handle_unknown == 'return_nan':
                X_unique.loc[-1] = np.nan
            elif self.handle_unknown == 'value':
                X_unique.loc[-1] = 0

            if self.handle_missing == 'return_nan':
                X_unique.loc[values.loc[np.nan]] = np.nan
            elif self.handle_missing == 'value':
                X_unique.loc[-2] = 0

            mappings_out.append({'col': col, 'mapping': X_unique})

        return mappings_out

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
            raise ValueError('Unexpected input dimension %d, expected %d' % (X.shape[1], self._dim,))

        if not list(self.cols):
            return X

        X_out = self.ordinal_encoder.transform(X)

        if self.handle_unknown == 'error':
            if X_out[self.cols].isin([-1]).any().any():
                raise ValueError('Columns to be encoded can not contain new values')

        X_out = self.basen_encode(X_out, cols=self.cols)

        if self.drop_invariant:
            for col in self.drop_cols:
                X_out.drop(col, 1, inplace=True)

        # impute missing values only in the generated columns
        # generated_cols = util.get_generated_cols(X, X_out, self.cols)
        # X_out[generated_cols] = X_out[generated_cols].fillna(value=0.0)

        if self.return_df or override_return_df:
            return X_out
        else:
            return X_out.values

    def inverse_transform(self, X_in):
        """
        Perform the inverse transformation to encoded data.

        Parameters
        ----------
        X_in : array-like, shape = [n_samples, n_features]

        Returns
        -------
        p: array, the same size of X_in

        """

        # fail fast
        if self._dim is None:
            raise ValueError('Must train encoder before it can be used to inverse_transform data')

        # unite the type into pandas dataframe (it makes the input size detection code easier...) and make deep copy
        X = util.convert_input(X_in, columns=self.feature_names, deep=True)

        X = self.basen_to_integer(X, self.cols, self.base)

        # make sure that it is the right size
        if X.shape[1] != self._dim:
            if self.drop_invariant:
                raise ValueError("Unexpected input dimension %d, the attribute drop_invariant should "
                                 "be False when transforming the data" % (X.shape[1],))
            else:
                raise ValueError('Unexpected input dimension %d, expected %d' % (X.shape[1], self._dim,))

        if not list(self.cols):
            return X if self.return_df else X.values

        for switch in self.ordinal_encoder.mapping:
            column_mapping = switch.get('mapping')
            inverse = pd.Series(data=column_mapping.index, index=column_mapping.values)
            X[switch.get('col')] = X[switch.get('col')].map(inverse).astype(switch.get('data_type'))

            if self.handle_unknown == 'return_nan' and self.handle_missing == 'return_nan':
                for col in self.cols:
                    if X[switch.get('col')].isnull().any():
                        warnings.warn("inverse_transform is not supported because transform impute "
                                      "the unknown category nan when encode %s" % (col,))

        return X if self.return_df else X.values

    def calc_required_digits(self, values):
        # figure out how many digits we need to represent the classes present
        if self.base == 1:
            digits = len(values) + 1
        else:
            digits = int(np.ceil(math.log(len(values), self.base))) + 1

        return digits

    def basen_encode(self, X_in, cols=None):
        """
        Basen encoding encodes the integers as basen code with one column per digit.

        Parameters
        ----------
        X_in: DataFrame
        cols: list-like, default None
            Column names in the DataFrame to be encoded

        Returns
        -------
        dummies : DataFrame

        """

        X = X_in.copy(deep=True)

        cols = X.columns.values.tolist()

        for switch in self.mapping:
            col = switch.get('col')
            mod = switch.get('mapping')

            base_df = mod.reindex(X[col])
            base_df.set_index(X.index, inplace=True)
            X = pd.concat([base_df, X], axis=1)

            old_column_index = cols.index(col)
            cols[old_column_index: old_column_index + 1] = mod.columns

        return X.reindex(columns=cols)

    def basen_to_integer(self, X, cols, base):
        """
        Convert basen code as integers.

        Parameters
        ----------
        X : DataFrame
            encoded data
        cols : list-like
            Column names in the DataFrame that be encoded
        base : int
            The base of transform

        Returns
        -------
        numerical: DataFrame

        """
        out_cols = X.columns.values.tolist()

        for col in cols:
            col_list = [col0 for col0 in out_cols if re.match(str(col)+'_\\d+', str(col0))]
            insert_at = out_cols.index(col_list[0])

            if base == 1:
                value_array = np.array([int(col0.split('_')[-1]) for col0 in col_list])
            else:
                len0 = len(col_list)
                value_array = np.array([base ** (len0 - 1 - i) for i in range(len0)])
            X.insert(insert_at, col, np.dot(X[col_list].values, value_array.T))
            X.drop(col_list, axis=1, inplace=True)
            out_cols = X.columns.values.tolist()

        return X

    def col_transform(self, col, digits):
        """
        The lambda body to transform the column values
        """

        if col is None or float(col) < 0.0:
            return None
        else:
            col = self.number_to_base(int(col), self.base, digits)
            if len(col) == digits:
                return col
            else:
                return [0 for _ in range(digits - len(col))] + col

    @staticmethod
    def number_to_base(n, b, limit):
        if b == 1:
            return [0 if n != _ else 1 for _ in range(limit)]

        if n == 0:
            return [0 for _ in range(limit)]

        digits = []
        for _ in range(limit):
            digits.append(int(n % b))
            n, _ = divmod(n, b)

        return digits[::-1]

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
            raise ValueError('Must fit data first. Affected feature names are not known before.')
        else:
            return self.feature_names
