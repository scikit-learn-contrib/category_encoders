"""BaseX encoding"""

import pandas as pd
import numpy as np
import math
import warnings
from sklearn.base import BaseEstimator, TransformerMixin
from category_encoders.ordinal import OrdinalEncoder
import category_encoders.utils as util

__author__ = 'willmcginnis'


class BaseNEncoder(BaseEstimator, TransformerMixin):
    """Base-N encoder encodes the categories into arrays of their base-N representation.  A base of 1 is equivalent to
    one-hot encoding (not really base-1, but useful), a base of 2 is equivalent to binary encoding. N=number of actual
    categories is equivalent to vanilla ordinal encoding.

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
    base: int
        when the downstream model copes well with nonlinearities (like decision tree), use higher base.
    impute_missing: bool
        boolean for whether or not to apply the logic for handle_unknown, will be deprecated in the future.
    handle_unknown: str
        options are 'error', 'ignore' and 'value', defaults to 'value'. Warning: if value is used,
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
    >>> enc = BaseNEncoder(cols=['CHAS', 'RAD']).fit(X, y)
    >>> numeric_dataset = enc.transform(X)
    >>> print(numeric_dataset.info())
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 506 entries, 0 to 505
    Data columns (total 18 columns):
    CHAS_0     506 non-null int64
    CHAS_1     506 non-null int64
    RAD_0      506 non-null int64
    RAD_1      506 non-null int64
    RAD_2      506 non-null int64
    RAD_3      506 non-null int64
    RAD_4      506 non-null int64
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
    dtypes: float64(11), int64(7)
    memory usage: 71.2 KB
    None

    """

    def __init__(self, verbose=0, cols=None, drop_invariant=False, return_df=True, base=2, impute_missing=True,
                 handle_unknown='value'):
        self.return_df = return_df
        self.drop_invariant = drop_invariant
        self.drop_cols = []
        self.verbose = verbose
        self.impute_missing = impute_missing
        self.handle_unknown = handle_unknown
        self.cols = cols
        self.ordinal_encoder = None
        self._dim = None
        self.base = base
        self._encoded_columns = None
        self.digits_per_col = {}

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

        # train an ordinal pre-encoder
        self.ordinal_encoder = OrdinalEncoder(
            verbose=self.verbose,
            cols=self.cols,
            handle_unknown=self.handle_unknown
        )
        self.ordinal_encoder = self.ordinal_encoder.fit(X)

        for col in self.cols:
            self.digits_per_col[col] = self.calc_required_digits(X, col)

        # do a transform on the training data to get a column list
        X_t = self.transform(X, override_return_df=True)
        self._encoded_columns = X_t.columns.values

        # drop all output columns with 0 variance.
        if self.drop_invariant:
            self.drop_cols = []
            X_temp = self.transform(X)
            generated_cols = util.get_generated_cols(X, X_temp, self.cols)
            self.drop_cols = [x for x in generated_cols if X_temp[x].var() <= 10e-5]

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

        if self._dim is None:
            raise ValueError('Must train encoder before it can be used to transform data.')

        # first check the type
        X = util.convert_input(X)

        # then make sure that it is the right size
        if X.shape[1] != self._dim:
            raise ValueError('Unexpected input dimension %d, expected %d' % (X.shape[1], self._dim,))

        if not self.cols:
            return X

        X_out = self.ordinal_encoder.transform(X)
        X_out = self.basen_encode(X_out, cols=self.cols)

        if self.drop_invariant:
            for col in self.drop_cols:
                X_out.drop(col, 1, inplace=True)

        # impute missing values only in the generated columns
        generated_cols = util.get_generated_cols(X, X_out, self.cols)
        X_out[generated_cols] = X_out[generated_cols].fillna(value=0.0)

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

        warnings.warn('Inverse transform in basen is a currently experimental feature, please be careful')
        X = X_in.copy(deep=True)

        # first check the type
        X = util.convert_input(X)

        if self._dim is None:
            raise ValueError('Must train encoder before it can be used to inverse_transform data')

        X = self.basen_to_integer(X, self.cols, self.base)

        # then make sure that it is the right size
        if X.shape[1] != self._dim:
            if self.drop_invariant:
                raise ValueError("Unexpected input dimension %d, the attribute drop_invariant should "
                                 "set as False when transform data" % (X.shape[1],))
            else:
                raise ValueError('Unexpected input dimension %d, expected %d' % (X.shape[1], self._dim,))

        if not self.cols:
            return X if self.return_df else X.values

        if self.impute_missing and self.handle_unknown == 'value':
            for col in self.cols:
                if any(X[col] == -1):
                    raise ValueError("inverse_transform is not supported because transform impute "
                                     "the unknown category -1 when encode %s" % (col,))

        for switch in self.ordinal_encoder.mapping:
            column_mapping = switch.get('mapping')
            inverse = pd.Series(data=column_mapping.index, index=column_mapping.get_values())
            X[switch.get('col')] = X[switch.get('col')].map(inverse).astype(switch.get('data_type'))

        return X if self.return_df else X.values

    def calc_required_digits(self, X, col):
        # figure out how many digits we need to represent the classes present
        if self.base == 1:
            digits = len(X[col].unique()) + 1
        else:
            digits = int(np.ceil(math.log(len(X[col].unique()), self.base))) + 1

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

        if cols is None:
            cols = X.columns.values
            pass_thru = []
        else:
            pass_thru = [col for col in X.columns.values if col not in cols]

        bin_cols = []
        for col in cols:
            # get how many digits we need to represent the classes present
            digits = self.calc_required_digits(X, col)

            # map the ordinal column into a list of these digits, of length digits
            X[col] = X[col].map(lambda x: self.col_transform(x, digits))

            for dig in range(digits):
                X[str(col) + '_%d' % (dig,)] = X[col].map(lambda r: int(r[dig]) if r is not None else None)
                bin_cols.append(str(col) + '_%d' % (dig,))

        if self._encoded_columns is None:
            X = X.reindex(columns=bin_cols + pass_thru)
        else:
            X = X.reindex(columns=self._encoded_columns)

        return X

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
        out_cols = X.columns.values

        for col in cols:
            col_list = [col0 for col0 in out_cols if str(col0).startswith(str(col))]
            for col0 in col_list:
                if any(X[col0].isnull()):
                    raise ValueError("inverse_transform is not supported because transform impute"
                                     "the unknown category -1 when encode %s" % (col,))
            if base == 1:
                value_array = np.array([int(col0.split('_')[-1]) for col0 in col_list])
            else:
                len0 = len(col_list)
                value_array = np.array([base ** (len0 - 1 - i) for i in range(len0)])

            X[col] = np.dot(X[col_list].values, value_array.T)
            out_cols = [col0 for col0 in out_cols if col0 not in col_list]

        X = X.reindex(columns=out_cols + cols)

        return X

    def col_transform(self, col, digits):
        """
        The lambda body to transform the column values
        """

        if col is None or float(col) < 0.0:
            return None
        else:
            col = self.numberToBase(int(col), self.base, digits)
            if len(col) == digits:
                return col
            else:
                return [0 for _ in range(digits - len(col))] + col

    @staticmethod
    def numberToBase(n, b, limit):
        if b == 1:
            return [0 if n != _ else 1 for _ in range(limit)]

        if n == 0:
            return [0 for _ in range(limit)]

        digits = []
        for _ in range(limit):
            digits.append(int(n % b))
            n, _ = divmod(n, b)

        return digits[::-1]
