"""BaseX encoding."""

import re
import warnings
from typing import Any

import numpy as np
import pandas as pd

import category_encoders.utils as util
from category_encoders.ordinal import OrdinalEncoder

__author__ = 'willmcginnis'


def _ceillogint(n, base):
    """
    Returns ceil(log(n, base)) for integers n and base.

    Uses integer math, so the result is not subject to floating point rounding errors.

    base must be >= 2 and n must be >= 1.
    """
    if base < 2:
        raise ValueError('base must be >= 2')
    if n < 1:
        raise ValueError('n must be >= 1')

    n -= 1
    ret = 0
    while n > 0:
        ret += 1
        n //= base
    return ret


class BaseNEncoder( util.UnsupervisedTransformerMixin,util.BaseEncoder):
    """Base-N encoder encodes the categories into arrays of their base-N representation.

    A base of 1 is equivalent to one-hot encoding (not really base-1, but useful),
    a base of 2 is equivalent to binary encoding.
    N=number of actual categories is equivalent to vanilla ordinal encoding.

    Parameters
    ----------
    verbose: int
        integer indicating verbosity of the output. 0 for none.
    cols: list
        a list of columns to encode, if None, all string columns will be encoded.
    drop_invariant: bool
        boolean for whether or not to drop columns with 0 variance.
    return_df: bool
        boolean for whether to return a pandas DataFrame from transform
        (otherwise it will be a numpy array).
    base: int
        when the downstream model copes well with nonlinearities (like decision tree),
        use higher base.
    handle_unknown: str
        options are 'error', 'return_nan', 'value', and 'indicator'. The default is 'value'.
        Warning: if indicator is used, an extra column will be added in if the transform matrix
        has unknown categories. This can cause unexpected changes in dimension in some cases.
    handle_missing: str
        options are 'error', 'return_nan', 'value', and 'indicator'. The default is 'value'.
        Warning: if indicator is used, an extra column will be added in if the transform matrix
        has nan values. This can cause unexpected changes in dimension in some cases.

    Example
    -------
    >>> from category_encoders import *
    >>> import pandas as pd
    >>> from sklearn.datasets import fetch_openml
    >>> bunch = fetch_openml(name='house_prices', as_frame=True)
    >>> display_cols = [
    ...     'Id',
    ...     'MSSubClass',
    ...     'MSZoning',
    ...     'LotFrontage',
    ...     'YearBuilt',
    ...     'Heating',
    ...     'CentralAir',
    ... ]
    >>> y = bunch.target
    >>> X = pd.DataFrame(bunch.data, columns=bunch.feature_names)[display_cols]
    >>> enc = BaseNEncoder(cols=['CentralAir', 'Heating']).fit(X, y)
    >>> numeric_dataset = enc.transform(X)
    >>> print(numeric_dataset.info())
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1460 entries, 0 to 1459
    Data columns (total 10 columns):
     #   Column        Non-Null Count  Dtype
    ---  ------        --------------  -----
     0   Id            1460 non-null   float64
     1   MSSubClass    1460 non-null   float64
     2   MSZoning      1460 non-null   object
     3   LotFrontage   1201 non-null   float64
     4   YearBuilt     1460 non-null   float64
     5   Heating_0     1460 non-null   int64
     6   Heating_1     1460 non-null   int64
     7   Heating_2     1460 non-null   int64
     8   CentralAir_0  1460 non-null   int64
     9   CentralAir_1  1460 non-null   int64
    dtypes: float64(4), int64(5), object(1)
    memory usage: 114.2+ KB
    None

    """

    prefit_ordinal = True
    encoding_relation = util.EncodingRelation.N_TO_M

    def __init__(
        self,
        verbose=0,
        cols=None,
        mapping=None,
        drop_invariant=False,
        return_df=True,
        base=2,
        handle_unknown='value',
        handle_missing='value',
    ):
        super().__init__(
            verbose=verbose,
            cols=cols,
            drop_invariant=drop_invariant,
            return_df=return_df,
            handle_unknown=handle_unknown,
            handle_missing=handle_missing,
        )
        self.mapping = mapping
        self.ordinal_encoder = None
        self.base = base

    def _fit(self, X, y=None, **kwargs):
        # train an ordinal pre-encoder
        self.ordinal_encoder = OrdinalEncoder(
            verbose=self.verbose, cols=self.cols, handle_unknown='value', handle_missing='value'
        )
        self.ordinal_encoder = self.ordinal_encoder.fit(X)

        self.mapping = self.fit_base_n_encoding()

    def fit_base_n_encoding(self) -> list[dict[str, Any]]:
        """Fit the base n encoder.

        Returns
        -------
        list[dict[str, Any]]
            List containing encoding mappings for each column.

        """
        mappings_out = []

        for switch in self.ordinal_encoder.category_mapping:
            col = switch.get('col')
            values = switch.get('mapping')

            if self.handle_missing == 'value':
                values = values[values > 0]

            if self.handle_unknown == 'indicator':
                values = np.append(values, -1)

            digits = self.calc_required_digits(values)
            X_unique = pd.DataFrame(
                index=values,
                columns=[f'{col}_{x}' for x in range(digits)],
                data=np.array([self.col_transform(x, digits) for x in range(1, len(values) + 1)]),
            )

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

    def _transform(self, X):
        X_out = self.ordinal_encoder.transform(X)

        if self.handle_unknown == 'error':
            if X_out[self.cols].isin([-1]).any().any():
                raise ValueError('Columns to be encoded can not contain new values')

        X_out = self.basen_encode(X_out, cols=self.cols)
        return X_out

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
       import pandas as pd

    if not isinstance(X, pd.DataFrame):
        raise ValueError("inverse_transform expects a pandas DataFrame as input.")

    #  NEW CHECK handle missing columns gracefully 
    expected_cols = getattr(self, "feature_names_in_", None)
    if expected_cols is not None:
        missing_cols = [c for c in expected_cols if c not in X.columns]
        if missing_cols:
            raise ValueError(f"Missing columns during inverse_transform: {missing_cols}")

    # Continue with existing dimension check
    if X.shape[1] != self._dim:
        raise ValueError(f"Unexpected input dimension {X.shape[1]}, expected {self._dim}")

    # Continue with rest of the logic
    X = X.copy()
    for switch in self.mapping:
        col = switch.get("col")
        if col in X:
            X[col] = X[col].map(switch.get("inverse_mapping"))
    return X
    def calc_required_digits(self, values: list) -> int:
        """Figure out how many digits we need to represent the classes present.

        Parameters
        ----------
        values: list
            list of values.

        Returns
        -------
        int
            number of digits necessary for encoding.
        """
        if self.base == 1:
            digits = len(values) + 1
        else:
            digits = _ceillogint(len(values) + 1, self.base)

        return digits

    def basen_encode(self, X_in: pd.DataFrame, cols=None):
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

        cols = X.columns.tolist()

        for switch in self.mapping:
            col = switch.get('col')
            mod = switch.get('mapping')

            base_df = mod.reindex(X[col])
            base_df = base_df.set_index(X.index)
            X = pd.concat([base_df, X], axis=1)

            old_column_index = cols.index(col)
            cols[old_column_index : old_column_index + 1] = mod.columns

        return X.reindex(columns=cols)

    def basen_to_integer(self, X: pd.DataFrame, cols, base):
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
        out_cols = X.columns.tolist()

        for col in cols:
            col_list = [
                col0 for col0 in out_cols if re.match(re.escape(str(col)) + '_\\d+', str(col0))
            ]
            insert_at = out_cols.index(col_list[0])

            if base == 1:
                value_array = np.array([int(col0.split('_')[-1]) for col0 in col_list])
            else:
                len0 = len(col_list)
                value_array = np.array([base ** (len0 - 1 - i) for i in range(len0)])
            X.insert(insert_at, col, np.dot(X[col_list].values, value_array.T))
            X = X.drop(col_list, axis=1)
            out_cols = X.columns.tolist()

        return X

    def col_transform(self, col, digits):
        """The lambda body to transform the column values."""
        if col is None or float(col) < 0.0:
            return None
        else:
            col = self.number_to_base(int(col), self.base, digits)
            if len(col) == digits:
                return col
            else:
                return [0 for _ in range(digits - len(col))] + col

    @staticmethod
    def number_to_base(n: int, b: int, limit: int) -> list[int]:
        """Convert number to base n representation (as list of digits).

        The list will be of length `limit`.

        Parameters
        ----------
        n: int
            number to convert
        b: int
            base
        limit: int
            length of representation.

        Returns
        -------
        list[int]
            base n representation as list of length limit containing the digits.
        """
        if b == 1:
            return [0 if n != _ else 1 for _ in range(limit)]

        if n == 0:
            return [0 for _ in range(limit)]

        digits = []
        for _ in range(limit):
            digits.append(int(n % b))
            n, _ = divmod(n, b)

        return digits[::-1]
