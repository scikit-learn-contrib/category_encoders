"""Gray encoding."""

from functools import partialmethod
from typing import List

import pandas as pd

from category_encoders import utils
from category_encoders.basen import BaseNEncoder

__author__ = 'paulwestenthanner'


class GrayEncoder(BaseNEncoder):
    """Gray encoding for categorical variables.

    Gray encoding is a form of binary encoding where consecutive values only differ by a single bit.
    Hence, gray encoding only makes sense for ordinal features.
    This has benefits in privacy preserving data publishing.

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
    >>> from category_encoders import GrayEncoder
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
    >>> enc = GrayEncoder(cols=['CentralAir', 'Heating']).fit(X, y)
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

    References
    ----------

    .. [1] https://en.wikipedia.org/wiki/Gray_code
    .. [2] Jun Zhang, Graham Cormode, Cecilia M. Procopiuc, Divesh Srivastava, and Xiaokui Xiao.
    2017. PrivBayes: Private Data Release via Bayesian Networks. ACM Trans. Database Syst. 42, 4,
    Article 25 (October 2017)
    """

    encoding_relation = utils.EncodingRelation.ONE_TO_M
    __init__ = partialmethod(BaseNEncoder.__init__, base=2)

    @staticmethod
    def gray_code(n: int, n_bit: int) -> List[int]:
        """Calculate the n-bit gray code for a value n.

        Parameters
        ----------
        n: int
            Value to encode (ordinal value of a category).
        n_bit: int
            Number of bits to encode to.

        Returns
        -------
        List[int]
            gray encoding of the input value.
        """
        gray = n ^ (n >> 1)
        gray_formatted = '{0:0{1}b}'.format(gray, n_bit)
        return [int(bit) for bit in gray_formatted]

    def _fit(self, X, y=None, **kwargs):
        super(GrayEncoder, self)._fit(X, y, **kwargs)
        gray_mapping = []
        # convert binary mapping to Gray mapping and reorder
        for col_to_encode in self.mapping:
            col = col_to_encode['col']
            bin_mapping = col_to_encode['mapping']
            n_cols_out = bin_mapping.shape[1]
            null_cond = (bin_mapping.index < 0) | (bin_mapping.isna().all(1))
            map_null = bin_mapping[null_cond]
            map_non_null = bin_mapping[~null_cond].copy()
            ordinal_mapping = [m for m in self.ordinal_encoder.mapping if m.get('col') == col]
            if len(ordinal_mapping) != 1:
                raise ValueError('Cannot find ordinal encoder mapping of Gray encoder')
            ordinal_mapping = ordinal_mapping[0]['mapping']
            reverse_ordinal_mapping = {v: k for k, v in ordinal_mapping.to_dict().items()}
            map_non_null['orig_value'] = map_non_null.index.to_series().map(reverse_ordinal_mapping)
            map_non_null = map_non_null.sort_values(by='orig_value')
            gray_encoding = [
                self.gray_code(i + 1, n_cols_out) for i in range(map_non_null.shape[0])
            ]
            gray_encoding = pd.DataFrame(
                data=gray_encoding, index=map_non_null.index, columns=bin_mapping.columns
            )
            gray_encoding = pd.concat([gray_encoding, map_null])
            gray_mapping.append({'col': col, 'mapping': gray_encoding})
        self.mapping = gray_mapping
