"""Binary encoding."""

from functools import partialmethod

from category_encoders import utils
from category_encoders.basen import BaseNEncoder

__author__ = 'willmcginnis'


class BinaryEncoder(BaseNEncoder):
    """Binary encoding for categorical variables.

    This is similar to onehot, but categories are stored as binary bitstrings.

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
    >>> enc = BinaryEncoder(cols=['CentralAir', 'Heating']).fit(X, y)
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

    encoding_relation = utils.EncodingRelation.ONE_TO_M
    __init__ = partialmethod(BaseNEncoder.__init__, base=2)
